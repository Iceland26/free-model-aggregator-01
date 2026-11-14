// free-model-aggregator-01: one server, three upstream wires, ten simulated free providers.
//
// A trial gives this server a brief ($MAKEFUN_PROMPT_FILE) and an env ($MAKEFUN_ENV_URL). The server
// probes each provider's catalog in its own wire, builds one catalog, registers it, then serves the
// client's OpenAI and Anthropic requests by routing each to the strongest free model that meets the
// request's requirements, speaking that provider's wire, and translating the answer back into the
// wire the client used.
//
//   - quality (60%): candidates are ranked by the sealed ratings' rating_milli, highest first.
//   - served (15%) & failover (10%): a first call to each provider is soft-capped so the load spreads
//     over many providers; a 402, 429 or 5xx fails over to the next candidate (one success per request).
//   - the wires: the three upstream request/event shapes and the two client answer shapes (JSON and
//     SSE for both) are in lib/translate.ts; lib/wire.ts holds the model lists and errors.
// Everything is deterministic: no clock, no randomness (the fixed `created` is in the client-shape
// helpers), so a replay sees the same bytes and the same provider requests.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import { anthropicError, anthropicModelList, openAIError, openAIModelList, type ServedModel } from './lib/wire.ts';
import { anthropicJSON, anthropicStream, buildUpstream, openAIJSON, openAIStream, parseProvider, type Candidate, type ProviderAnswer } from './lib/translate.ts';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;

/** How many first-attempt calls one provider may take before the router moves down the ladder. */
const SOFT_CAPS: Record<string, number> = {
  openrouter: 8,
  gemini: 5,
  cloudflare: 5,
  groq: 6,
  pollinations: 5,
  cerebras: 4,
  ovh: 4,
  kilol: 8,
  sambanova: 3,
  zai: 3,
};

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
let manifestIndex = 0;
/** How many upstream calls each provider has taken so far (first attempts and failovers alike). */
const attempts: Record<string, number> = {};

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered: exactly what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  for (const m of out) m.id = m.id;
  return out;
}

/** Every free (provider, model) meeting `requires`, ranked best first. */
function candidatesFor(requires: Requirement): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({
        providerId: probe.provider.id,
        wire: probe.provider.wire,
        auth: probe.provider.auth,
        base: probe.provider.base,
        keyEnv: probe.provider.key_env,
        key: probe.provider.key,
        modelId: model.id,
        rating: row?.rating_milli ?? model.rating_milli ?? 0,
      });
    }
  }
  out.sort((a, b) => b.rating - a.rating || (a.providerId < b.providerId ? -1 : a.providerId > b.providerId ? 1 : 0) || (a.modelId < b.modelId ? -1 : a.modelId > b.modelId ? 1 : 0));
  return out;
}

/** The requirements of the request now arriving: the manifest's row, else the body's own signals. */
function requirementsFor(body: Json): Requirement {
  const entry = brief.workload?.manifest?.[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  sendJson(res, status, wire === 'anthropic' ? anthropicError(type, message) : openAIError(message, type, null));
}

function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  sendJson(res, 200, typeof req.headers['x-api-key'] === 'string' ? anthropicModelList(list) : openAIModelList(list));
}

/** Whether a provider status is a refusal to fail over from (402, 429, or any 5xx). */
function isRefusal(status: number): boolean {
  return status === 402 || status === 429 || status >= 500;
}

/** POST one request upstream in the candidate's wire; returns the HTTP status and the whole body text. */
async function postUpstream(c: Candidate, body: Json, stream: boolean): Promise<{ status: number; text: string }> {
  const base = joinBase(ENV_URL, c.base);
  let url: string;
  if (c.wire === 'openai') url = `${base}/v1/chat/completions`;
  else if (c.wire === 'anthropic') url = `${base}/v1/messages`;
  else url = `${base}/v1beta/models/${c.modelId}:${stream ? 'streamGenerateContent' : 'generateContent'}`;
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(c.auth, c.key) };
  if (c.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  if (stream) headers.accept = 'text/event-stream';
  const res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(body) });
  return { status: res.status, text: await res.text() };
}

/** Answer the client with one normalized provider answer, in the client's wire and stream shape. */
function respond(res: ServerResponse, api: 'openai' | 'anthropic', stream: boolean, answer: ProviderAnswer, model: string): void {
  if (stream) {
    const text: string = api === 'anthropic' ? anthropicStream(answer, model) : openAIStream(answer, model);
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', 'content-length': Buffer.byteLength(text) });
    res.end(text);
    return;
  }
  sendJson(res, 200, api === 'anthropic' ? anthropicJSON(answer, model) : openAIJSON(answer, model));
}

/** Serve one client request by trying candidates best-first, failing over on 402/429/5xx. */
async function serve(req: IncomingMessage, res: ServerResponse, api: 'openai' | 'anthropic', body: Json): Promise<void> {
  const requires = requirementsFor(body);
  const list = candidatesFor(requires);
  if (list.length === 0) {
    return sendError(res, api, 503, api === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory meets this request');
  }
  const stream = body.stream === true;

  // First attempt: the best-rated candidate whose provider is still under its soft cap.
  let first: Candidate | null = null;
  for (const c of list) {
    if ((attempts[c.providerId] ?? 0) < (SOFT_CAPS[c.providerId] ?? Number.POSITIVE_INFINITY)) {
      first = c;
      break;
    }
  }
  if (first === null) first = list[0];
  const order: Candidate[] = [first, ...list];

  const tried = new Set<string>();
  for (const c of order) {
    const key = `${c.providerId}/${c.modelId}`;
    if (tried.has(key)) continue;
    tried.add(key);
    attempts[c.providerId] = (attempts[c.providerId] ?? 0) + 1;
    let upstream: { status: number; text: string };
    try {
      upstream = await postUpstream(c, buildUpstream(c, api, body), stream);
    } catch (err) {
      log(`upstream ${c.providerId}/${c.modelId} failed: ${(err as Error).message}`);
      continue;
    }
    if (upstream.status >= 200 && upstream.status < 300) {
      const answer = parseProvider(c.wire, stream, upstream.text);
      return respond(res, api, stream, answer, c.modelId);
    }
    if (isRefusal(upstream.status)) continue; // 402/429/5xx: try the next candidate.
    // 401/400/404 mean this entry's own request is wrong, not a budget refusal: surface it.
    return sendError(res, api, upstream.status, api === 'anthropic' ? 'api_error' : 'invalid_request_error', `provider ${c.providerId} answered HTTP ${upstream.status}: ${upstream.text.slice(0, 300)}`);
  }
  sendError(res, api, api === 'anthropic' ? 529 : 502, api === 'anthropic' ? 'overloaded_error' : 'upstream_unreachable', 'every candidate provider refused this request');
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && (url.pathname === '/v1/chat/completions' || url.pathname === '/v1/messages')) {
    const api = url.pathname === '/v1/messages' ? 'anthropic' : 'openai';
    const body = await readJson(req);
    if (body === null) return sendError(res, api, 400, 'invalid_request_error', 'the request body is not one JSON object');
    await serve(req, res, api, body);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}