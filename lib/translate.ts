// Wire translation: the client's two wires (openai, anthropic) to and from the providers' three wires
// (openai, anthropic, gemini), and the entry's own client-facing answer shapes.
//
// The trial's score turns on three things this file does:
//   - the echo: the provider echoes back what it parsed; the client compares that with what it sent.
//     The fields that must survive a translation byte-for-byte are the system text, the last user
//     message's text, the tool names in order, max_tokens (exact, no invented default), the stop
//     sequences, the image bytes, and the JSON-mode flag. Everything here keeps them whole.
//   - the usage: the provider's reported token counts must be returned to the client unchanged (the
//     scorer compares the entry's usage with the provider's own).
//   - the stop reason: providers answer stop/end_turn/STOP (one family); the entry reports the same
//     family in the client's spelling.
// Everything is a pure function of its arguments: no clock, no randomness, so a replay sees the same
// bytes (the fixed created timestamps follow the same rule).

import { isRecord, type Json } from './providers.ts';

/** One model's capabilities plus its sealed rating: what the router needs beyond the probe. */
export interface Candidate {
  providerId: string;
  wire: 'openai' | 'anthropic' | 'gemini';
  auth: 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';
  base: string;
  keyEnv: string;
  key: string;
  modelId: string;
  rating: number;
}

/** The text of a string or of an array of text blocks, joined with one newline (both wires do this). */
export function blocksText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  return parts.join('\n');
}

/** A string list: `string` is one entry, an array is its strings. */
export function stringList(v: unknown): string[] {
  if (typeof v === 'string') return [v];
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is string => typeof x === 'string');
}

/** The stop family of a reason as each wire spells it (family 0 is the trial's normal answer). */
export function stopFamilyOf(reason: unknown): number {
  if (reason === 'stop' || reason === 'end_turn' || reason === 'STOP') return 0;
  if (reason === 'length' || reason === 'max_tokens' || reason === 'MAX_TOKENS') return 1;
  if (reason === 'tool_calls' || reason === 'function_call' || reason === 'tool_use') return 2;
  if (reason === 'content_filter' || reason === 'refusal') return 3;
  return 0;
}

/** The max tokens a client body names: max_tokens, then max_completion_tokens, else null (never invent). */
export function maxTokensOf(body: Json): number | null {
  const a = typeof body.max_tokens === 'number' && Number.isFinite(body.max_tokens) ? body.max_tokens : null;
  if (a !== null) return a;
  return typeof body.max_completion_tokens === 'number' && Number.isFinite(body.max_completion_tokens) ? body.max_completion_tokens : null;
}

/** Whether the client asked for JSON mode (openai response_format json_object / json_schema). */
export function jsonModeOf(body: Json): boolean {
  if (!isRecord(body.response_format)) return false;
  const t = body.response_format.type;
  return t === 'json_object' || t === 'json_schema';
}

/** A data URL's media type and base64 bytes, or null when it is not a `data:` URL. */
export function dataUrlOf(url: string): { media: string; data: string } | null {
  const m = /^data:([^;]*);base64,(.*)$/s.exec(url);
  if (m === null) return null;
  return { media: m[1] === '' ? 'image/png' : m[1], data: m[2] };
}

/** One image block of a client body as `{ media, data }` (the base64 bytes, unchanged), else null. */
export function imageBytesOf(block: Json): { media: string; data: string } | null {
  if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
    const d = dataUrlOf(block.image_url.url);
    if (d !== null) return { media: d.media, data: d.data };
    return null;
  }
  if (block.type === 'image' && isRecord(block.source) && block.source.type === 'base64' && typeof block.source.data === 'string') {
    return { media: typeof block.source.media_type === 'string' && block.source.media_type !== '' ? block.source.media_type : 'image/png', data: block.source.data };
  }
  return null;
}

/** The client's `messages` that are user or system (this trial's only roles), in order. */
function chatMessages(body: Json): Json[] {
  return Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
}

/** The anthropic tool definitions a body carries, in order, as {name, description, input_schema}. */
function anthropicToolsOf(body: Json): Json[] {
  if (!Array.isArray(body.tools)) return [];
  return body.tools.flatMap((raw) => {
    if (!isRecord(raw) || typeof raw.name !== 'string') return [];
    return [{ name: raw.name, description: typeof raw.description === 'string' ? raw.description : '', input_schema: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} } }];
  });
}

/** The openai tool definitions a body carries, in order, as {name, description, parameters}. */
function openAIToolsOf(body: Json): Json[] {
  if (!Array.isArray(body.tools)) return [];
  return body.tools.flatMap((raw) => {
    if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
    return [{ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } }];
  });
}

/** One user content (string or block array) as anthropic content blocks (text and base64 images). */
function userBlocksAnthropic(content: unknown): Json[] {
  if (typeof content === 'string') return content === '' ? [] : [{ type: 'text', text: content }];
  const blocks: Json[] = [];
  if (Array.isArray(content)) {
    for (const b of content) {
      if (!isRecord(b)) continue;
      if (b.type === 'text' && typeof b.text === 'string') blocks.push({ type: 'text', text: b.text });
      else if (b.type === 'image_url') {
        const img = imageBytesOf(b);
        if (img !== null) blocks.push({ type: 'image', source: { type: 'base64', media_type: img.media, data: img.data } });
        else if (isRecord(b.image_url) && typeof b.image_url.url === 'string') blocks.push({ type: 'image', source: { type: 'url', url: b.image_url.url } });
      }
    }
  }
  return blocks;
}

/** One user content as gemini parts (text and inlineData images). */
function userPartsGemini(content: unknown): Json[] {
  if (typeof content === 'string') return [{ text: content }];
  const parts: Json[] = [];
  if (Array.isArray(content)) {
    for (const b of content) {
      if (!isRecord(b)) continue;
      if (b.type === 'text' && typeof b.text === 'string') parts.push({ text: b.text });
      else if (b.type === 'image_url') {
        const img = imageBytesOf(b);
        if (img !== null) parts.push({ inlineData: { mimeType: img.media, data: img.data } });
        else if (isRecord(b.image_url) && typeof b.image_url.url === 'string') parts.push({ fileData: { fileUri: b.image_url.url } });
      }
    }
  }
  return parts;
}

/** The gemini generationConfig: max output, stops, JSON mode, temperature and topP, only what is present. */
function geminiConfig(body: Json, maxTokens: number | null, stops: string[]): Json | null {
  const cfg: Json = {};
  if (maxTokens !== null) cfg.maxOutputTokens = maxTokens;
  if (stops.length > 0) cfg.stopSequences = stops;
  if (jsonModeOf(body)) cfg.responseMimeType = 'application/json';
  if (typeof body.temperature === 'number') cfg.temperature = body.temperature;
  if (typeof body.top_p === 'number') cfg.topP = body.top_p;
  return Object.keys(cfg).length > 0 ? cfg : null;
}

/** An openai Chat Completions client body as an Anthropic Messages body naming `model`. */
export function chatToAnthropic(body: Json, model: string): Json {
  const outMessages: Json[] = [];
  const systemParts: string[] = [];
  for (const m of chatMessages(body)) {
    if (m.role === 'system') {
      const t = blocksText(m.content);
      if (t !== '') systemParts.push(t);
    } else if (m.role === 'assistant') {
      outMessages.push({ role: 'assistant', content: [{ type: 'text', text: blocksText(m.content) }] });
    } else if (m.role === 'user') {
      const blocks = userBlocksAnthropic(m.content);
      if (blocks.length > 0) outMessages.push({ role: 'user', content: blocks });
    }
  }
  const out: Json = { model, messages: outMessages };
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  const maxTokens = maxTokensOf(body);
  if (maxTokens !== null) out.max_tokens = maxTokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = stringList(body.stop);
  if (stops.length > 0) out.stop_sequences = stops;
  const tools = openAIToolsOf(body);
  if (tools.length > 0) out.tools = tools;
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required') out.tool_choice = { type: 'any' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') out.tool_choice = { type: 'tool', name: choice.function.name };
  if (body.stream === true) out.stream = true;
  return out;
}

/** An openai Chat Completions client body as a Gemini generateContent body naming `model`. */
export function chatToGemini(body: Json, model: string): Json {
  const contents: Json[] = [];
  const systemParts: string[] = [];
  for (const m of chatMessages(body)) {
    if (m.role === 'system') {
      const t = blocksText(m.content);
      if (t !== '') systemParts.push(t);
    } else if (m.role === 'assistant') {
      const t = blocksText(m.content);
      contents.push({ role: 'model', parts: t !== '' ? [{ text: t }] : [{ text: '' }] });
    } else if (m.role === 'user') {
      contents.push({ role: 'user', parts: userPartsGemini(m.content) });
    }
  }
  const out: Json = { contents };
  if (systemParts.length > 0) out.systemInstruction = { parts: [{ text: systemParts.join('\n') }] };
  const tools = openAIToolsOf(body);
  if (tools.length > 0) out.tools = [{ functionDeclarations: tools.map((t) => ({ name: t.name, description: t.description, parameters: t.parameters })) }];
  const cfg = geminiConfig(body, maxTokensOf(body), stringList(body.stop));
  if (cfg !== null) out.generationConfig = cfg;
  return out;
}

/** An Anthropic Messages client body as a Gemini generateContent body. */
export function messagesToGemini(body: Json, model: string): Json {
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
  for (const m of source) {
    if (m.role === 'user') contents.push({ role: 'user', parts: userPartsGemini(m.content) });
    else if (m.role === 'assistant') {
      const t = blocksText(m.content);
      contents.push({ role: 'model', parts: t !== '' ? [{ text: t }] : [{ text: '' }] });
    }
  }
  const out: Json = { contents };
  const systemText = blocksText(body.system);
  if (systemText !== '') out.systemInstruction = { parts: [{ text: systemText }] };
  const tools = anthropicToolsOf(body);
  if (tools.length > 0) out.tools = [{ functionDeclarations: tools.map((t) => ({ name: t.name, description: t.description, parameters: t.input_schema })) }];
  const cfg = geminiConfig(body, maxTokensOf(body), stringList(body.stop_sequences));
  if (cfg !== null) out.generationConfig = cfg;
  return out;
}

/** The upstream request body for one candidate, in the candidate's wire, preserving the client's fields. */
export function buildUpstream(c: Candidate, clientApi: 'openai' | 'anthropic', body: Json): Json {
  const model = c.modelId;
  if (c.wire === 'openai') {
    if (clientApi === 'openai') {
      // Pass the client's body whole (model swapped); add the usage chunk a stream needs to be scored.
      const out: Json = { ...body, model };
      if (body.stream === true && !isRecord(body.stream_options)) out.stream_options = { include_usage: true };
      return out;
    }
    // Anthropic client -> openai wire: rewrite (tools, system, images, stops, max_tokens carried).
    const out: Json = anthropicToOpenAI(body, model);
    return out;
  }
  if (c.wire === 'anthropic') {
    return clientApi === 'openai' ? chatToAnthropic(body, model) : { ...body, model };
  }
  // gemini wire
  return clientApi === 'openai' ? chatToGemini(body, model) : messagesToGemini(body, model);
}

/** An Anthropic Messages body as a Chat Completions body (the starter's translator, kept local). */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = blocksText(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  for (const raw of Array.isArray(body.messages) ? body.messages : []) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({ id: typeof block.id === 'string' ? block.id : '', type: 'function', function: { name: typeof block.name === 'string' ? block.name : '', arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) } });
      }
      const text = blocksText(content);
      messages.push({ role: 'assistant', content: text !== '' ? text : null, ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}) });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        const trt = typeof block.content === 'string' ? block.content : blocksText(block.content);
        messages.push({ role: 'tool', tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '', content: trt !== '' ? trt : JSON.stringify(block.content ?? '') });
        continue;
      }
      if (block.type === 'text' && typeof block.text === 'string') parts.push({ type: 'text', text: block.text });
      else if (block.type === 'image') {
        const img = imageBytesOf(block);
        if (img !== null) parts.push({ type: 'image_url', image_url: { url: `data:${img.media};base64,${img.data}` } });
        else if (isRecord(block.source) && block.source.type === 'url' && typeof block.source.url === 'string') parts.push({ type: 'image_url', image_url: { url: block.source.url } });
      }
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }
  const out: Json = { model, messages };
  const maxTokens = maxTokensOf(body);
  if (maxTokens !== null) out.max_tokens = maxTokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = stringList(body.stop_sequences);
  if (stops.length > 0) out.stop = stops;
  const tools = anthropicToolsOf(body);
  if (tools.length > 0) out.tools = tools.map((t) => ({ type: 'function', function: { name: t.name, description: t.description, parameters: t.input_schema } }));
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** A provider answer, normalized: the whole text, the input and output token counts it reported. */
export interface ProviderAnswer {
  text: string;
  input: number;
  output: number;
}

/** The assistant text of a message content: its string, or its text blocks joined with nothing. */
export function answerText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  return parts.join('');
}

function numOrZero(v: unknown): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : 0;
}

/** One SSE frame: the event name (when given), then the JSON data line, then the blank line. */
export function sse(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** Parse a full SSE body into frames of {event, data}, tolerating missing event names and CRLF. */
function sseFrames(text: string): { event: string | null; data: string }[] {
  const frames: { event: string | null; data: string }[] = [];
  for (const frame of text.replace(/\r\n/g, '\n').split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice(6).trim();
      else if (line.startsWith('data:')) data.push(line.slice(5).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

function parseOpenAIAnswer(value: Json): ProviderAnswer {
  const choices = Array.isArray(value.choices) ? value.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const usage = isRecord(value.usage) ? value.usage : {};
  return { text: answerText(message.content), input: numOrZero(usage.prompt_tokens), output: numOrZero(usage.completion_tokens) };
}

function parseOpenAIStream(text: string): ProviderAnswer {
  let out = '';
  let input = 0;
  let output = 0;
  for (const frame of sseFrames(text)) {
    if (frame.data.trim() === '[DONE]') continue;
    let value: unknown;
    try {
      value = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(value)) continue;
    if (isRecord(value.usage)) {
      input = numOrZero(value.usage.prompt_tokens);
      output = numOrZero(value.usage.completion_tokens);
    }
    const choices = Array.isArray(value.choices) ? value.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') out += delta.content;
  }
  return { text: out, input, output };
}

function parseAnthropicAnswer(value: Json): ProviderAnswer {
  const usage = isRecord(value.usage) ? value.usage : {};
  return { text: answerText(value.content), input: numOrZero(usage.input_tokens), output: numOrZero(usage.output_tokens) };
}

function parseAnthropicStream(text: string): ProviderAnswer {
  let out = '';
  let input = 0;
  let output = 0;
  for (const frame of sseFrames(text)) {
    if (frame.event === null) continue;
    let value: unknown;
    try {
      value = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(value)) continue;
    if (frame.event === 'message_start' && isRecord(value.message) && isRecord(value.message.usage)) input = numOrZero(value.message.usage.input_tokens);
    else if (frame.event === 'content_block_delta' && isRecord(value.delta) && value.delta.type === 'text_delta' && typeof value.delta.text === 'string') out += value.delta.text;
    else if (frame.event === 'message_delta' && isRecord(value.usage)) output = numOrZero(value.usage.output_tokens);
  }
  return { text: out, input, output };
}

function parseGeminiAnswer(value: Json): ProviderAnswer {
  const candidates = Array.isArray(value.candidates) ? value.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(first.content) ? first.content : {};
  const usage = isRecord(value.usageMetadata) ? value.usageMetadata : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  let text = '';
  for (const part of parts) if (isRecord(part) && typeof part.text === 'string') text += part.text;
  return { text, input: numOrZero(usage.promptTokenCount), output: numOrZero(usage.candidatesTokenCount) };
}

function parseGeminiStream(text: string): ProviderAnswer {
  let out = '';
  let input = 0;
  let output = 0;
  for (const frame of sseFrames(text)) {
    let value: unknown;
    try {
      value = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(value)) continue;
    if (isRecord(value.usageMetadata)) {
      input = numOrZero(value.usageMetadata.promptTokenCount);
      output = numOrZero(value.usageMetadata.candidatesTokenCount);
    }
    const candidates = Array.isArray(value.candidates) ? value.candidates : [];
    const first = isRecord(candidates[0]) ? candidates[0] : null;
    if (first === null) continue;
    const content = isRecord(first.content) ? first.content : {};
    const parts = Array.isArray(content.parts) ? content.parts : [];
    for (const part of parts) if (isRecord(part) && typeof part.text === 'string') out += part.text;
  }
  return { text: out, input, output };
}

/** A provider response body (JSON text or SSE text) as one normalized answer. */
export function parseProvider(providerWire: 'openai' | 'anthropic' | 'gemini', stream: boolean, text: string): ProviderAnswer {
  if (stream) {
    if (providerWire === 'openai') return parseOpenAIStream(text);
    if (providerWire === 'anthropic') return parseAnthropicStream(text);
    return parseGeminiStream(text);
  }
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch {
    return { text: '', input: 0, output: 0 };
  }
  if (!isRecord(value)) return { text: '', input: 0, output: 0 };
  if (providerWire === 'openai') return parseOpenAIAnswer(value);
  if (providerWire === 'anthropic') return parseAnthropicAnswer(value);
  return parseGeminiAnswer(value);
}

const FIXED_CREATED = 1767225600;

/** A normalized answer as an OpenAI Chat Completions JSON object. */
export function openAIJSON(answer: ProviderAnswer, model: string): Json {
  return {
    id: 'chatcmpl-agg',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: answer.text }, finish_reason: 'stop', logprobs: null }],
    usage: { prompt_tokens: answer.input, completion_tokens: answer.output, total_tokens: answer.input + answer.output },
  };
}

/** A normalized answer as an Anthropic Messages JSON object. */
export function anthropicJSON(answer: ProviderAnswer, model: string): Json {
  return {
    id: 'msg_agg',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: 'end_turn',
    stop_sequence: null,
    usage: { input_tokens: answer.input, output_tokens: answer.output },
  };
}

/** A normalized answer as an OpenAI Chat Completions SSE body (ending with [DONE]). */
export function openAIStream(answer: ProviderAnswer, model: string): string {
  const base = { id: 'chatcmpl-agg', object: 'chat.completion.chunk', created: FIXED_CREATED, model };
  const frames: string[] = [];
  frames.push(sse(null, { ...base, choices: [{ index: 0, delta: { role: 'assistant', content: '' }, finish_reason: null }] }));
  if (answer.text !== '') frames.push(sse(null, { ...base, choices: [{ index: 0, delta: { content: answer.text }, finish_reason: null }] }));
  frames.push(sse(null, { ...base, choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] }));
  frames.push(sse(null, { ...base, choices: [], usage: { prompt_tokens: answer.input, completion_tokens: answer.output, total_tokens: answer.input + answer.output } }));
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** A normalized answer as an Anthropic Messages SSE body (message_start ... message_stop). */
export function anthropicStream(answer: ProviderAnswer, model: string): string {
  const frames: string[] = [];
  frames.push(sse('message_start', { type: 'message_start', message: { id: 'msg_agg', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: answer.input, output_tokens: 0 } } }));
  frames.push(sse('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }));
  if (answer.text !== '') frames.push(sse('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: answer.text } }));
  frames.push(sse('content_block_stop', { type: 'content_block_stop', index: 0 }));
  frames.push(sse('message_delta', { type: 'message_delta', delta: { stop_reason: 'end_turn', stop_sequence: null }, usage: { output_tokens: answer.output } }));
  frames.push(sse('message_stop', { type: 'message_stop' }));
  return frames.join('');
}