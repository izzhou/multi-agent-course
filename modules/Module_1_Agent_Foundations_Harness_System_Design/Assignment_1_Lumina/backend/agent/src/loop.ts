/**
 * The quick loop: plan, choose a tool, observe, repeat, then answer.
 *
 * Two phases, on purpose:
 *   1. research. The model calls web_search and fetch_page until it has enough, under the
 *      gear's caps. Each call is a `trace` event the moment it finishes.
 *   2. answer. `sources` goes out first (every page actually fetched, numbered), then the
 *      answer streams token by token from a call that sees only those pages.
 * Splitting them is what lets `sources` precede the first token without guessing, and it
 * means the text the user reads was written from fetched pages, not search snippets.
 *
 * Fail loud: an LLM exception is a ProviderError and the caller turns it into a 502. A
 * failed tool is a trace step with ok:false that the model sees and can route around; if
 * every search failed and nothing was read, that is an outage, not an empty answer.
 */
import type Anthropic from '@anthropic-ai/sdk';
import { z } from 'zod';
import {
  newId,
  unresolvedCitations,
  type AskMode,
  type DoneEvent,
  type Locator,
  type RunLog,
  type Source,
  type Terminated,
  type TraceEvent
} from '@lumina/contract';
import { env } from './env.js';
import { anthropic, addUsage, baseParams, llmCostUsd, ProviderError, type Usage } from './llm.js';
import { webSearch } from './search.js';
import { bestPassage, fetchPage, prefetch } from './fetchPage.js';
import { embeddingCostUsd } from './embeddings.js';
import { recallMemories, saveMemory } from './memory.js';
import { searchDocuments, spaceHasIndexedDocs } from './rag.js';

export type Emit = (event: 'trace' | 'sources' | 'token', data: unknown) => void;

/** Filled in as the run goes, so a run that throws can still be logged honestly. */
export interface RunState {
  started: number;
  usage: Usage;
  toolCalls: RunLog['toolCalls'];
  embedTokens: number;
}

export const newRunState = (): RunState => ({
  started: Date.now(),
  usage: { in: 0, out: 0 },
  toolCalls: [],
  embedTokens: 0
});

export const runCostUsd = (s: RunState, uncachedSearches: number) =>
  llmCostUsd(s.usage) + embeddingCostUsd(s.embedTokens) + uncachedSearches * env.searchUsdPerCall;

export interface AskInput {
  userId: string;
  threadId: string;
  query: string;
  mode: AskMode;
  spaceId?: string;
  history: { role: 'user' | 'assistant'; content: string }[];
  emit: Emit;
  state: RunState;
}

export interface AskOutput {
  text: string;
  sources: Source[];
  done: DoneEvent;
  runLog: RunLog;
}

/** Something retrieved in this request: a fetched page or a document chunk. The only things an [n] may point at. */
interface Citable {
  n: number;
  kind: 'web' | 'doc';
  title: string;
  url?: string;
  docId?: string;
  locator?: Locator;
  /** Full text the answer is written from. */
  text: string;
  /** For a doc, the top-ranked chunk's own text, verbatim. Web snippets are picked at the end. */
  snippet?: string;
}

const locatorLabel = (l?: Locator) =>
  l?.page ? `p. ${l.page}` : l?.heading ? `"${l.heading}"` : l?.line ? `line ${l.line}` : '';

const RESEARCH_EXCERPT_CHARS = 2500;

const TOOLS: Anthropic.Tool[] = [
  {
    name: 'web_search',
    description:
      'Search the web. Returns titles, URLs and short snippets. Snippets are leads, not evidence: fetch the pages you intend to rely on.',
    input_schema: {
      type: 'object',
      properties: {
        query: { type: 'string', description: 'The search query.' },
        reason: { type: 'string', description: 'One short line: why this search.' }
      },
      required: ['query', 'reason']
    }
  },
  {
    name: 'fetch_page',
    description:
      'Fetch a web page and read its text. The page becomes a numbered source the answer can cite.',
    input_schema: {
      type: 'object',
      properties: {
        url: { type: 'string', description: 'An http(s) URL, usually from web_search results.' },
        reason: { type: 'string', description: 'One short line: why this page.' }
      },
      required: ['url', 'reason']
    }
  },
  {
    name: 'save_memory',
    description:
      'Save one durable fact or preference about the user, for all future conversations. Only for things the user states about themselves or explicitly asks you to remember, never for facts learned from a search.',
    input_schema: {
      type: 'object',
      properties: {
        text: {
          type: 'string',
          description: 'The fact or preference as one self-contained sentence, e.g. "Prefers TypeScript code examples over prose."'
        },
        reason: { type: 'string', description: 'One short line: why this is worth remembering.' }
      },
      required: ['text', 'reason']
    }
  }
];

const DOCS_TOOL: Anthropic.Tool = {
  name: 'search_documents',
  description:
    "Search the user's uploaded documents in this Space (hybrid keyword and semantic search). Returns numbered passages with their page or section.",
  input_schema: {
    type: 'object',
    properties: {
      query: { type: 'string', description: 'What to look for in the documents.' },
      reason: { type: 'string', description: 'One short line: why search the documents.' }
    },
    required: ['query', 'reason']
  }
};

const SaveInput = z.object({ text: z.string().min(3).max(500), reason: z.string().optional() });

const SearchInput = z.object({ query: z.string().min(1), reason: z.string().optional() });
const FetchInput = z.object({ url: z.string().url(), reason: z.string().optional() });

const RESEARCH_SYSTEM = `You research questions for a search engine that answers with citations.
When a search_documents tool is available, the user's own documents come first: if passages already shown answer the question, you are done. Use the web only for what the documents do not cover.
Otherwise use web_search, then fetch_page on the 2 to 4 most promising results. Call several tools in one turn when they are independent.
Prefer primary and authoritative sources. Do not fetch a page you already fetched.
When the fetched pages are enough to answer, stop calling tools and reply with the single word READY. Do not write the answer yourself.
Keep it cheap: most questions need one search and two or three pages.
If the user states a lasting preference or fact about themselves, or asks you to remember something, call save_memory once with it. If the message is only that request, you do not need to search.`;

const ANSWER_SYSTEM = `You write the answer for a search engine, using ONLY the numbered sources provided.
Rules:
- Put a citation like [1] or [2][3] right after every claim that a source supports. Use only the source numbers given.
- Never invent facts, URLs or source numbers. If the sources do not answer the question, say so plainly and cite nothing.
- Lead with the direct answer. Be concise: a short paragraph or a few bullets, no preamble.
- Follow the user's saved preferences when they are given. They change how you answer, never what the sources say.`;

const describeError = (err: unknown) => (err instanceof Error ? err.message : String(err)) || 'unknown error';

/**
 * Drops any [n] that does not name a source of this request, even across token
 * boundaries. The prompt asks for grounded citations; this makes it a guarantee.
 */
export class CitationFilter {
  private pending = '';
  constructor(private readonly valid: Set<number>) {}
  push(chunk: string): string {
    let out = '';
    for (const ch of chunk) {
      if (this.pending) {
        this.pending += ch;
        if (ch === ']') {
          const n = Number(this.pending.slice(1, -1));
          if (this.valid.has(n)) out += this.pending;
          this.pending = '';
        } else if (!/[0-9]/.test(ch) || this.pending.length > 5) {
          out += this.pending;
          this.pending = '';
        }
      } else if (ch === '[') {
        this.pending = '[';
      } else {
        out += ch;
      }
    }
    return out;
  }
  flush(): string {
    const rest = this.pending;
    this.pending = '';
    return rest;
  }
}

export async function runQuick(input: AskInput): Promise<AskOutput> {
  const { started, usage, toolCalls } = input.state;
  const state = input.state;
  const caps = { toolCalls: env.maxToolCalls, deadline: started + env.maxWallClockSec * 1000 };
  const fetched: Citable[] = [];
  const docSearches = { total: 0, failed: 0, errors: [] as string[] };
  const useDocs = !!input.spaceId && input.mode !== 'web';
  const useWeb = input.mode !== 'docs';
  const tools = [...(useWeb ? TOOLS.slice(0, 2) : []), ...(useDocs ? [DOCS_TOOL] : []), TOOLS[2]!];
  const searches = { total: 0, cached: 0, failed: 0, errors: [] as string[] };
  let step = 0;
  let terminated: Terminated = 'done';

  const historyText = input.history.length
    ? `Earlier in this conversation:\n${input.history
        .map((m) => `${m.role}: ${m.content.slice(0, 1200)}`)
        .join('\n')}\n\n`
    : '';

  // ------------------------------------------------------------ tools

  /** One step of the loop: a trace event the moment it finishes, and a line in the run log. */
  const tracer = (tool: string, raw: Record<string, unknown>) => {
    const t0 = Date.now();
    const thisStep = ++step;
    return (ok: boolean, error?: string) => {
      const ev: TraceEvent = {
        step: thisStep,
        tool: tool as TraceEvent['tool'],
        input: raw,
        ok,
        ms: Date.now() - t0,
        ...(typeof raw.reason === 'string' ? { reason: raw.reason } : {}),
        ...(error ? { error } : {})
      };
      input.emit('trace', ev);
      toolCalls.push({ name: ev.tool, ok, ms: ev.ms, ...(error ? { error } : {}) });
    };
  };

  const saved: string[] = [];

  /** Runs a document search, registers each hit as a citable source, and returns the text the model sees. */
  async function docSearch(query: string): Promise<string> {
    docSearches.total++;
    try {
      const { hits, embedTokens } = await searchDocuments(input.userId, input.spaceId!, query);
      state.embedTokens += embedTokens;
      if (!hits.length) return 'No matching passages in this Space.';
      const shown: string[] = [];
      for (const h of hits) {
        // One source per document and locator: two chunks of the same page are one citation.
        let src = fetched.find((f) => f.kind === 'doc' && f.docId === h.docId && locatorLabel(f.locator) === locatorLabel(h.locator));
        if (!src) {
          src = { n: fetched.length + 1, kind: 'doc', title: h.title, docId: h.docId, locator: h.locator, text: h.text, snippet: h.text };
          fetched.push(src);
        } else if (!src.text.includes(h.text)) {
          src.text += `\n${h.text}`;
        }
        shown.push(`Source [${src.n}]: ${h.title}, ${locatorLabel(h.locator)}\n${h.text}`);
      }
      return shown.join('\n\n');
    } catch (err) {
      docSearches.failed++;
      docSearches.errors.push(describeError(err));
      throw err;
    }
  }

  async function runTool(use: Anthropic.ToolUseBlock): Promise<Anthropic.ToolResultBlockParam> {
    const raw = (use.input ?? {}) as Record<string, unknown>;
    const trace = tracer(use.name, raw);

    try {
      if (use.name === 'web_search') {
        const { query } = SearchInput.parse(raw);
        searches.total++;
        try {
          const { results, cached } = await webSearch(query);
          if (cached) searches.cached++;
          prefetch(results.slice(0, 3).map((r) => r.url));
          trace(true);
          const list = results.length
            ? results.map((r, i) => `${i + 1}. ${r.title}\n   ${r.url}\n   ${r.snippet.slice(0, 300)}`).join('\n')
            : 'No results.';
          return { type: 'tool_result', tool_use_id: use.id, content: list };
        } catch (err) {
          searches.failed++;
          searches.errors.push(describeError(err));
          throw err;
        }
      }

      if (use.name === 'fetch_page') {
        const { url } = FetchInput.parse(raw);
        const seen = fetched.find((f) => f.url === url);
        const page = seen ?? (await fetchPage(url));
        const src = seen ?? { n: fetched.length + 1, kind: 'web' as const, title: page.title, url, text: page.text };
        if (!seen) fetched.push(src);
        trace(true);
        return {
          type: 'tool_result',
          tool_use_id: use.id,
          content: `Source [${src.n}]: ${src.title} (${src.url})\n\n${src.text.slice(0, RESEARCH_EXCERPT_CHARS)}`
        };
      }

      if (use.name === 'search_documents' && useDocs) {
        const { query } = SearchInput.parse(raw);
        const content = await docSearch(query);
        trace(true);
        return { type: 'tool_result', tool_use_id: use.id, content };
      }

      if (use.name === 'save_memory') {
        const { text } = SaveInput.parse(raw);
        const { id, duplicate, embedTokens } = await saveMemory(input.userId, text, input.threadId);
        state.embedTokens += embedTokens;
        saved.push(text);
        trace(true);
        return {
          type: 'tool_result',
          tool_use_id: use.id,
          content: duplicate ? `Already remembered (${id}).` : `Saved as ${id}. The user can see and delete it in their Memory panel.`
        };
      }

      throw new Error(`unknown tool ${use.name}`);
    } catch (err) {
      const error = describeError(err);
      trace(false, error);
      return { type: 'tool_result', tool_use_id: use.id, content: `Error: ${error}`, is_error: true };
    }
  }

  // ------------------------------------------------------------ recall

  // Every answer starts by recalling the user's saved memories: it is one embedding and one
  // indexed query, and leaving it to the model would make a saved preference apply only
  // when the model happens to think of looking.
  let memoryText = '';
  {
    const raw = { query: input.query, reason: "recall the user's saved preferences before planning" };
    const trace = tracer('recall_memory', raw);
    try {
      const { memories, embedTokens } = await recallMemories(input.userId, input.query);
      state.embedTokens += embedTokens;
      trace(true);
      if (memories.length) {
        memoryText = `The user's saved preferences and facts:\n${memories.map((m) => `- ${m.text}`).join('\n')}\n\n`;
      }
    } catch (err) {
      trace(false, describeError(err));
    }
  }

  // ------------------------------------------------------------ router

  // A Space with indexed documents is searched first, with the question itself, so a
  // corpus question never depends on the model remembering to look.
  let docsText = '';
  if (useDocs && (input.mode === 'docs' || (await spaceHasIndexedDocs(input.userId, input.spaceId!)))) {
    const raw = {
      query: input.query,
      reason:
        input.mode === 'docs'
          ? 'mode is docs: answer from this Space'
          : "router: this Space has indexed documents, so check them before the web"
    };
    const trace = tracer('search_documents', raw);
    try {
      docsText = `Passages already retrieved from the user's documents:\n\n${await docSearch(input.query)}\n\n`;
      trace(true);
    } catch (err) {
      trace(false, describeError(err));
    }
  }

  // ------------------------------------------------------------ phase 1: research

  const messages: Anthropic.MessageParam[] = [
    { role: 'user', content: `${memoryText}${historyText}${docsText}Question: ${input.query}` }
  ];

  for (;;) {
    const remainingMs = caps.deadline - Date.now();
    if (remainingMs <= 0 || toolCalls.length >= caps.toolCalls) {
      terminated = 'cap';
      break;
    }
    let resp: Anthropic.Message;
    try {
      resp = await anthropic.messages.create(
        { ...baseParams, max_tokens: 4096, system: RESEARCH_SYSTEM, tools, messages },
        { signal: AbortSignal.timeout(remainingMs) }
      );
    } catch (err) {
      if (Date.now() >= caps.deadline) {
        terminated = 'cap';
        break;
      }
      throw new ProviderError(`llm: ${describeError(err)}`);
    }
    addUsage(usage, resp.usage);

    const uses = resp.content.filter((b): b is Anthropic.ToolUseBlock => b.type === 'tool_use');
    if (resp.stop_reason !== 'tool_use' || uses.length === 0) break;

    const budget = caps.toolCalls - toolCalls.length;
    const results = await Promise.all(uses.slice(0, budget).map(runTool));
    for (const skipped of uses.slice(budget)) {
      results.push({
        type: 'tool_result',
        tool_use_id: skipped.id,
        content: 'Not run: the tool-call cap for this search was reached.',
        is_error: true
      });
    }
    messages.push({ role: 'assistant', content: resp.content }, { role: 'user', content: results });
    if (uses.length > budget) {
      terminated = 'cap';
      break;
    }
  }

  const attempts = searches.total + docSearches.total;
  if (attempts > 0 && searches.failed + docSearches.failed === attempts && fetched.length === 0) {
    // Every retrieval threw and nothing was read: that is an outage, never "no results".
    throw new ProviderError(`retrieval failed: ${[...searches.errors, ...docSearches.errors][0]}`);
  }

  // ------------------------------------------------------------ phase 2: answer

  const sources: Source[] = fetched.map((f) =>
    f.kind === 'doc'
      ? { n: f.n, kind: 'doc', title: f.title, docId: f.docId!, ...(f.locator ? { locator: f.locator } : {}), snippet: f.snippet! }
      : { n: f.n, kind: 'web', title: f.title, url: f.url!, snippet: bestPassage(f.text, input.query) || f.text.slice(0, 300) }
  );
  input.emit('sources', sources);

  const sourceBlock = fetched.length
    ? fetched
        .map((f) =>
          f.kind === 'doc'
            ? `[${f.n}] ${f.title}, ${locatorLabel(f.locator)} (the user's document)\n${f.text}`
            : `[${f.n}] ${f.title} (${f.url})\n${f.text.slice(0, env.pageMaxChars)}`
        )
        .join('\n\n---\n\n')
    : 'No sources were retrieved.';
  const savedNote = saved.length
    ? `\nThis turn you saved to the user's long-term memory: ${saved.map((t) => `"${t}"`).join('; ')}. Confirm that in one sentence.`
    : '';
  const capNote =
    terminated === 'cap'
      ? '\nResearch was cut short by the time or tool-call cap. Say briefly that the answer may be incomplete.'
      : '';

  const filter = new CitationFilter(new Set(fetched.map((f) => f.n)));
  let text = '';
  let ttftMs = 0;
  const out = (piece: string) => {
    if (!piece) return;
    if (!ttftMs) ttftMs = Date.now() - started;
    text += piece;
    input.emit('token', { text: piece });
  };

  try {
    const stream = anthropic.messages.stream({
      ...baseParams,
      max_tokens: 2048,
      system: ANSWER_SYSTEM + savedNote + capNote,
      messages: [
        {
          role: 'user',
          content: `${memoryText}${historyText}Sources:\n\n${sourceBlock}\n\nQuestion: ${input.query}`
        }
      ]
    });
    stream.on('text', (delta) => out(filter.push(delta)));
    const final = await stream.finalMessage();
    out(filter.flush());
    addUsage(usage, final.usage);
  } catch (err) {
    throw new ProviderError(`llm: ${describeError(err)}`);
  }

  // The filter makes this impossible; checking anyway is what makes it a guarantee.
  const dangling = unresolvedCitations(text, sources);
  if (dangling.length) throw new Error(`ungrounded citations ${dangling.join(', ')}`);

  const latencyMs = Date.now() - started;
  const uncachedSearches = searches.total - searches.cached;
  const costUsd = runCostUsd(state, uncachedSearches);

  const done: DoneEvent = {
    answerId: newId('ans'),
    latencyMs,
    ttftMs: ttftMs || latencyMs,
    model: env.llmModel,
    tokens: { in: usage.in, out: usage.out },
    costUsd: Number(costUsd.toFixed(6)),
    searchCached: searches.total > 0 && searches.cached === searches.total,
    terminated,
    depth: 'quick',
    subQuestions: 0
  };

  const runLog: RunLog = {
    tokens: usage.in + usage.out,
    wallClockSec: Number((latencyMs / 1000).toFixed(3)),
    costUsd: done.costUsd,
    terminated,
    depth: 'quick',
    toolCalls
  };

  return { text, sources, done, runLog };
}
