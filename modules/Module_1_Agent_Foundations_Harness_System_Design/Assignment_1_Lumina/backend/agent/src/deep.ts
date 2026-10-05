/**
 * Deep search: plan, fan out, merge, answer.
 *
 *   1. plan_research. One short structured call splits the question into 3 to 6
 *      sub-questions, and the `plan` event goes out before anything is retrieved.
 *   2. fan out. Every sub-question is researched at once: a search, then the best few
 *      pages nobody else has claimed. Every retrieval step and every source carries the
 *      number of the sub-question it served.
 *   3. merge. One numbering across all sub-questions, grouped by sub-question, so [3] means
 *      the same page everywhere and a reader can see why each source is in the list.
 *   4. answer. Streamed like a quick answer, from those sources only, in a fixed shape:
 *      the direct answer, a section per sub-question, then what is still unknown.
 *
 * The research step is code, not a model loop. The plan already says what to look for, so
 * a second model deciding which tool to call next would add latency and cost and buy
 * nothing; it also makes the spend predictable: the tool-call cap is shared out up front.
 */
import { z } from 'zod';
import {
  newId,
  unresolvedCitations,
  type DoneEvent,
  type PlanEvent,
  type RunLog,
  type Source,
  type SubQuestion,
  type Terminated,
  type TraceEvent
} from '@lumina/contract';
import { env } from './env.js';
import { anthropic, addUsage, baseParams, ProviderError } from './llm.js';
import { webSearch, type SearchResult } from './search.js';
import { bestPassage, excerptAround, fetchPage } from './fetchPage.js';
import { recallMemories } from './memory.js';
import { searchDocuments, spaceHasIndexedDocs } from './rag.js';
import {
  CitationFilter,
  describeError,
  locatorLabel,
  runCostUsd,
  type AskInput,
  type AskOutput,
  type Citable
} from './loop.js';

/** A sub-question plus the search query the planner wrote for it. */
interface Planned extends SubQuestion {
  query: string;
}

/** A citable source, plus the sub-question that found it and the order it was found in. */
interface Found extends Citable {
  subQuestion: number;
  seq: number;
}

const PLAN_SCHEMA = {
  type: 'object',
  properties: {
    reason: { type: 'string', description: 'One sentence: how the question breaks down.' },
    subQuestions: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          question: { type: 'string', description: 'A self-contained sub-question.' },
          query: { type: 'string', description: 'A web search query for it, 3 to 10 words.' },
          reason: { type: 'string', description: 'One short line: why the answer needs this.' }
        },
        required: ['question', 'query', 'reason'],
        additionalProperties: false
      }
    }
  },
  required: ['reason', 'subQuestions'],
  additionalProperties: false
} as const;

const PlanOut = z.object({
  reason: z.string(),
  subQuestions: z.array(z.object({ question: z.string().min(1), query: z.string().min(1), reason: z.string() }))
});

const planSystem = () => `You plan research for a search engine. Split the user's question into ${env.deepSubQuestionsMin} to ${env.deepSubQuestionsMax} sub-questions that together answer it.
Each sub-question must be self-contained (no "it" or "they" that needs the original question), must cover a different part of the question, and must be something a web search can answer.
Order them so the ones the answer depends on most come first. Use the fewest sub-questions that cover the whole question.`;

const ANSWER_SYSTEM = `You write the answer for a deep research search engine, using ONLY the numbered sources provided. They are grouped by the sub-question that found them, but any source may support any claim.
Shape the answer exactly like this, in Markdown:
1. The direct answer to the whole question in two to four sentences, first, with no heading.
2. One section per sub-question, in the order given, headed "## " followed by a short title for it. Each section is a short paragraph or a few bullets.
3. A final section "## What is still unknown": what the sources did not settle, disagree on, or may have out of date. One to three bullets.
Rules:
- Put a citation like [1] or [2][3] right after every claim that a source supports. Use only the source numbers given.
- Never invent facts, URLs or source numbers. If no source covers a sub-question, say so in its section and cite nothing.
- Follow the user's saved preferences when they are given. They change how you answer, never what the sources say.`;

export async function runDeep(input: AskInput): Promise<AskOutput> {
  const state = input.state;
  const { started, usage, toolCalls } = state;
  const deadline = started + env.maxWallClockSecDeep * 1000;
  const useDocs = !!input.spaceId && input.mode !== 'web';
  const useWeb = input.mode !== 'docs';
  const found: Found[] = [];
  const claimed = new Set<string>();
  const retrieval = { total: 0, failed: 0, errors: [] as string[] };
  const searches = { total: 0, cached: 0 };
  let step = 0;
  let reserved = 0;
  // Widened on purpose: reserve() sets it from inside closures, which narrowing cannot see.
  let terminated = 'done' as Terminated;

  const historyText = input.history.length
    ? `Earlier in this conversation:\n${input.history.map((m) => `${m.role}: ${m.content.slice(0, 800)}`).join('\n')}\n\n`
    : '';

  /** Takes one tool call from the budget, or says the cap is reached. */
  const reserve = () => {
    if (reserved >= env.maxToolCallsDeep || Date.now() >= deadline) {
      terminated = 'cap';
      return false;
    }
    reserved++;
    return true;
  };

  const tracer = (tool: TraceEvent['tool'], raw: Record<string, unknown>, subQuestion?: number) => {
    const t0 = Date.now();
    const thisStep = ++step;
    return (ok: boolean, error?: string) => {
      const ev: TraceEvent = {
        step: thisStep,
        tool,
        input: raw,
        ok,
        ms: Date.now() - t0,
        ...(typeof raw.reason === 'string' ? { reason: raw.reason } : {}),
        ...(error ? { error } : {}),
        ...(subQuestion ? { subQuestion } : {})
      };
      input.emit('trace', ev);
      toolCalls.push({ name: tool, ok, ms: ev.ms, ...(error ? { error } : {}) });
    };
  };

  // ------------------------------------------------------------ plan (and recall, alongside)

  // Recall is not retrieval, so it may run while the planner thinks: the plan is the deep
  // search's first paint, and nothing should wait in front of it.
  const recall = (async () => {
    if (!reserve()) return '';
    const trace = tracer('recall_memory', { query: input.query, reason: "recall the user's saved preferences" });
    try {
      const { memories, embedTokens } = await recallMemories(input.userId, input.query);
      state.embedTokens += embedTokens;
      trace(true);
      return memories.length ? `The user's saved preferences and facts:\n${memories.map((m) => `- ${m.text}`).join('\n')}\n\n` : '';
    } catch (err) {
      trace(false, describeError(err));
      return '';
    }
  })();

  reserve();
  const planTrace = tracer('plan_research', { query: input.query, reason: 'split the question into sub-questions' });
  let plan: Planned[];
  let planReason = '';
  try {
    const ask = async (nudge: string) => {
      const resp = await anthropic.messages.create(
        {
          model: env.deepPlanModel,
          max_tokens: 1024,
          output_config: { effort: env.llmEffort, format: { type: 'json_schema', schema: PLAN_SCHEMA } },
          system: planSystem(),
          messages: [{ role: 'user', content: `${historyText}Question: ${input.query}${nudge}` }]
        },
        { signal: AbortSignal.timeout(30_000) }
      );
      addUsage(usage, resp.usage);
      const text = resp.content.map((b) => (b.type === 'text' ? b.text : '')).join('');
      return PlanOut.parse(JSON.parse(text));
    };
    let out = await ask('');
    if (out.subQuestions.length < env.deepSubQuestionsMin) {
      // Too few sub-questions is a quick search in disguise. One retry, then fail loud.
      out = await ask(`\n\nYou returned ${out.subQuestions.length} sub-questions. Return at least ${env.deepSubQuestionsMin}.`);
    }
    if (out.subQuestions.length < Math.max(2, env.deepSubQuestionsMin)) {
      throw new Error(`planner returned ${out.subQuestions.length} sub-questions, needs at least ${env.deepSubQuestionsMin}`);
    }
    planReason = out.reason;
    plan = out.subQuestions.slice(0, env.deepSubQuestionsMax).map((s, k) => ({ i: k + 1, ...s }));
  } catch (err) {
    planTrace(false, describeError(err));
    throw new ProviderError(`plan_research: ${describeError(err)}`);
  }
  const planEvent: PlanEvent = {
    subQuestions: plan.map(({ i, question, reason }) => ({ i, question, ...(reason ? { reason } : {}) })),
    ...(planReason ? { reason: planReason } : {})
  };
  input.emit('plan', planEvent);
  planTrace(true);

  const memoryText = await recall;

  // ------------------------------------------------------------ fan out

  const docsToo = useDocs && (input.mode === 'docs' || (await spaceHasIndexedDocs(input.userId, input.spaceId!)));
  const perSub = Math.floor((env.maxToolCallsDeep - reserved) / plan.length);
  const pagesPerSub = useWeb ? Math.max(1, Math.min(env.deepPagesPerSubQuestion, perSub - 1 - (docsToo ? 1 : 0))) : 0;

  const addFound = (src: Omit<Found, 'n' | 'seq'>) => {
    const f: Found = { ...src, n: 0, seq: found.length };
    found.push(f);
    return f;
  };

  async function docsFor(sq: Planned): Promise<void> {
    if (!reserve()) return;
    const trace = tracer('search_documents', { query: sq.question, reason: `documents for sub-question ${sq.i}` }, sq.i);
    retrieval.total++;
    try {
      const { hits, embedTokens } = await searchDocuments(input.userId, input.spaceId!, sq.question);
      state.embedTokens += embedTokens;
      for (const h of hits) {
        const key = `${h.docId}:${locatorLabel(h.locator)}`;
        if (claimed.has(key)) continue;
        claimed.add(key);
        addFound({ kind: 'doc', title: h.title, docId: h.docId, locator: h.locator, text: h.text, snippet: h.text, subQuestion: sq.i });
      }
      trace(true);
    } catch (err) {
      retrieval.failed++;
      retrieval.errors.push(describeError(err));
      trace(false, describeError(err));
    }
  }

  async function read(sq: Planned, r: SearchResult): Promise<boolean> {
    if (!reserve()) return false;
    const trace = tracer('fetch_page', { url: r.url, reason: `read for sub-question ${sq.i}: ${r.title.slice(0, 80)}` }, sq.i);
    try {
      const page = await fetchPage(r.url);
      addFound({ kind: 'web', title: page.title, url: r.url, text: page.text, subQuestion: sq.i });
      trace(true);
      return true;
    } catch (err) {
      trace(false, describeError(err));
      return false;
    }
  }

  async function webFor(sq: Planned): Promise<void> {
    if (!reserve()) return;
    const trace = tracer('web_search', { query: sq.query, reason: `search for sub-question ${sq.i}` }, sq.i);
    retrieval.total++;
    searches.total++;
    let results: SearchResult[];
    try {
      const out = await webSearch(sq.query);
      if (out.cached) searches.cached++;
      results = out.results;
      trace(true);
    } catch (err) {
      retrieval.failed++;
      retrieval.errors.push(describeError(err));
      trace(false, describeError(err));
      return;
    }
    // Claim pages synchronously, so two sub-questions never spend calls reading the same page.
    const queue = results.filter((r) => !claimed.has(r.url));
    let ok = 0;
    while (ok < pagesPerSub && queue.length) {
      const batch = queue.splice(0, pagesPerSub - ok).filter((r) => !claimed.has(r.url));
      batch.forEach((r) => claimed.add(r.url));
      const got = await Promise.all(batch.map((r) => read(sq, r)));
      ok += got.filter(Boolean).length;
      // A failed page is replaced from the same results, only while the budget allows.
      if (got.some((g) => !g) && reserved >= env.maxToolCallsDeep) break;
    }
  }

  await Promise.all(
    plan.map(async (sq) => {
      if (docsToo) await docsFor(sq);
      if (useWeb) await webFor(sq);
    })
  );

  if (retrieval.total > 0 && retrieval.failed === retrieval.total && found.length === 0) {
    throw new ProviderError(`retrieval failed: ${retrieval.errors[0]}`);
  }

  // ------------------------------------------------------------ merge

  // Number by sub-question, then by discovery order, so the list reads in plan order.
  found.sort((a, b) => a.subQuestion - b.subQuestion || a.seq - b.seq);
  found.forEach((f, k) => (f.n = k + 1));
  const byN = new Map(plan.map((sq) => [sq.i, sq]));
  const focus = (f: Found) => `${byN.get(f.subQuestion)?.question ?? ''} ${input.query}`;

  const sources: Source[] = found.map((f) =>
    f.kind === 'doc'
      ? {
          n: f.n,
          kind: 'doc',
          title: f.title,
          docId: f.docId!,
          ...(f.locator ? { locator: f.locator } : {}),
          snippet: f.snippet!,
          subQuestion: f.subQuestion
        }
      : {
          n: f.n,
          kind: 'web',
          title: f.title,
          url: f.url!,
          snippet: bestPassage(f.text, focus(f)) || f.text.slice(0, 300),
          subQuestion: f.subQuestion
        }
  );
  input.emit('sources', sources);

  // ------------------------------------------------------------ answer

  const perSource = Math.min(env.pageMaxChars, Math.floor(env.deepSourceCharsTotal / Math.max(1, found.length)));
  const sourceBlock = plan
    .map((sq) => {
      const mine = found.filter((f) => f.subQuestion === sq.i);
      const body = mine.length
        ? mine
            .map((f) =>
              f.kind === 'doc'
                ? `[${f.n}] ${f.title}, ${locatorLabel(f.locator)} (the user's document)\n${f.text}`
                : `[${f.n}] ${f.title} (${f.url})\n${excerptAround(f.text, focus(f), perSource)}`
            )
            .join('\n\n')
        : 'No sources were found for this sub-question.';
      return `### Sub-question ${sq.i}: ${sq.question}\n\n${body}`;
    })
    .join('\n\n---\n\n');
  const capNote =
    terminated === 'cap'
      ? '\nResearch was cut short by the time or tool-call cap. Say so in "What is still unknown".'
      : '';

  const filter = new CitationFilter(new Set(found.map((f) => f.n)));
  let text = '';
  let ttftMs = 0;
  const out = (piece: string) => {
    if (!piece) return;
    if (!ttftMs) ttftMs = Date.now() - started;
    text += piece;
    input.emit('token', { text: piece });
  };

  try {
    const stream = anthropic.messages.stream(
      {
        ...baseParams,
        max_tokens: 3000,
        system: ANSWER_SYSTEM + capNote,
        messages: [
          {
            role: 'user',
            content: `${memoryText}${historyText}Sources:\n\n${sourceBlock}\n\nQuestion: ${input.query}\n\nSub-questions, in order:\n${plan
              .map((sq) => `${sq.i}. ${sq.question}`)
              .join('\n')}`
          }
        ]
      },
      { signal: AbortSignal.timeout(Math.max(1000, deadline - Date.now())) }
    );
    stream.on('text', (delta) => out(filter.push(delta)));
    const final = await stream.finalMessage();
    out(filter.flush());
    addUsage(usage, final.usage);
  } catch (err) {
    throw new ProviderError(`llm: ${describeError(err)}`);
  }

  const dangling = unresolvedCitations(text, sources);
  if (dangling.length) throw new Error(`ungrounded citations ${dangling.join(', ')}`);

  const latencyMs = Date.now() - started;
  const costUsd = runCostUsd(state, searches.total - searches.cached);

  const done: DoneEvent = {
    answerId: newId('ans'),
    latencyMs,
    ttftMs: ttftMs || latencyMs,
    model: env.llmModel,
    tokens: { in: usage.in, out: usage.out },
    costUsd: Number(costUsd.toFixed(6)),
    searchCached: searches.total > 0 && searches.cached === searches.total,
    terminated,
    depth: 'deep',
    subQuestions: plan.length
  };

  const runLog: RunLog = {
    tokens: usage.in + usage.out,
    wallClockSec: Number((latencyMs / 1000).toFixed(3)),
    costUsd: done.costUsd,
    terminated,
    depth: 'deep',
    toolCalls
  };

  return { text, sources, done, runLog, plan: planEvent.subQuestions };
}
