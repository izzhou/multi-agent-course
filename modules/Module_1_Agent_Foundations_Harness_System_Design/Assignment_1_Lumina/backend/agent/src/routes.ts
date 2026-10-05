/**
 * Threads and the ask route. Every route here trusts nothing about the caller: the
 * gateway checks X-User-Id, and so does this service, because it is reachable directly.
 */
import express from 'express';
import type { Logger } from 'pino';
import { randomUUID } from 'node:crypto';
import { writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import {
  AskBody,
  COLLECTIONS,
  CreateThreadBody,
  ThreadId,
  REQUEST_HEADER,
  USER_HEADER,
  newId,
  type GetThreadResponse,
  type ListMemoryResponse,
  type ListThreadsResponse,
  type MessageDoc,
  type RunDoc,
  type RunLog,
  type ThreadDoc
} from '@lumina/contract';
import { env } from './env.js';
import { db } from './db.js';
import { sseHeaders, sseSend } from './sse.js';
import { newRunState, runCostUsd, runQuick } from './loop.js';
import { ProviderError } from './llm.js';
import { runDeep } from './deep.js';
import { refundDeep, resetsAt, takeDeep } from './deepCap.js';
import { deleteMemory, listMemories } from './memory.js';
import { ownSpace, registerSpaceRoutes } from './spaces.js';

const HISTORY_MESSAGES = 6;

type Req = express.Request;
type Res = express.Response;

const requestIdOf = (req: Req, res: Res): string => {
  const id = (req.header(REQUEST_HEADER) ?? `req_${randomUUID().slice(0, 12)}`).trim();
  res.setHeader(REQUEST_HEADER, id);
  return id;
};

const fail = (res: Res, status: number, error: string, requestId?: string) =>
  res.status(status).json({ error, status, ...(requestId ? { requestId } : {}) });

const iso = (d: Date | string) => new Date(d).toISOString();

/** The caller's thread, or null. Another user's thread is a 404, never a 403: ids do not leak. */
async function ownThread(threadId: string, userId: string) {
  if (!ThreadId.safeParse(threadId).success) return null;
  return (await db()).collection<ThreadDoc>(COLLECTIONS.threads).findOne({ _id: threadId, userId });
}

export function registerRoutes(app: express.Express, log: Logger): void {
  // X-User-Id on every route but /health.
  app.use((req, res, next) => {
    if (req.path === '/health') return next();
    const userId = req.header(USER_HEADER)?.trim();
    if (!userId) return fail(res, 401, 'X-User-Id header is required');
    res.locals.userId = userId;
    next();
  });

  // ---------------------------------------------------------------- threads

  app.post('/threads', async (req, res, next) => {
    try {
      const body = CreateThreadBody.safeParse(req.body ?? {});
      if (!body.success) return fail(res, 400, body.error.issues[0]?.message ?? 'invalid body');
      const doc: ThreadDoc = {
        _id: newId('thr'),
        userId: res.locals.userId,
        title: body.data.title ?? 'New thread',
        createdAt: new Date()
      };
      await (await db()).collection<ThreadDoc>(COLLECTIONS.threads).insertOne(doc);
      res.status(201).json({ threadId: doc._id });
    } catch (err) {
      next(err);
    }
  });

  app.get('/threads', async (_req, res, next) => {
    try {
      const rows = await (await db())
        .collection<ThreadDoc>(COLLECTIONS.threads)
        .find({ userId: res.locals.userId })
        .sort({ createdAt: -1 })
        .limit(100)
        .toArray();
      const body: ListThreadsResponse = {
        threads: rows.map((t) => ({ threadId: t._id, title: t.title, createdAt: iso(t.createdAt) }))
      };
      res.json(body);
    } catch (err) {
      next(err);
    }
  });

  app.get('/threads/:threadId', async (req, res, next) => {
    try {
      const thread = await ownThread(req.params.threadId, res.locals.userId);
      if (!thread) return fail(res, 404, `no thread ${req.params.threadId}`);
      const rows = await (await db())
        .collection<MessageDoc>(COLLECTIONS.messages)
        .find({ threadId: thread._id })
        .sort({ createdAt: 1 })
        .toArray();
      const body: GetThreadResponse = {
        threadId: thread._id,
        title: thread.title,
        messages: rows.map((m) => ({
          role: m.role,
          content: m.content,
          sources: m.sources,
          ...(m.answerId ? { answerId: m.answerId } : {}),
          ...(m.done ? { done: m.done } : {}),
          createdAt: iso(m.createdAt)
        }))
      };
      res.json(body);
    } catch (err) {
      next(err);
    }
  });

  registerSpaceRoutes(app);

  // ---------------------------------------------------------------- memory

  app.get('/memory', async (_req, res, next) => {
    try {
      const body: ListMemoryResponse = { memories: await listMemories(res.locals.userId) };
      res.json(body);
    } catch (err) {
      next(err);
    }
  });

  app.delete('/memory/:memoryId', async (req, res, next) => {
    try {
      const gone = await deleteMemory(res.locals.userId, req.params.memoryId);
      if (!gone) return fail(res, 404, `no memory ${req.params.memoryId}`);
      res.status(204).end();
    } catch (err) {
      next(err);
    }
  });

  // ---------------------------------------------------------------- ask

  app.post('/threads/:threadId/ask', async (req, res) => {
    const requestId = requestIdOf(req, res);
    const userId: string = res.locals.userId;
    const state = newRunState();

    const parsed = AskBody.safeParse(req.body ?? {});
    if (!parsed.success) return fail(res, 400, parsed.error.issues[0]?.message ?? 'invalid body', requestId);
    const ask = parsed.data;

    if (ask.mode === 'docs' && !ask.spaceId) return fail(res, 400, 'mode "docs" needs a spaceId', requestId);

    let streaming = false;
    const emit = (event: string, data: unknown) => {
      if (!streaming) {
        sseHeaders(res);
        streaming = true;
      }
      sseSend(res, event, data);
    };

    const writeRun = async (runLog: RunLog, extra: Partial<RunDoc>) => {
      await writeFile(join(env.runsDir, `${requestId}.json`), JSON.stringify(runLog, null, 2));
      await (await db())
        .collection<RunDoc>(COLLECTIONS.runs)
        .insertOne({ ...runLog, requestId, userId, threadId: req.params.threadId, query: ask.query, createdAt: new Date(), ...extra });
    };

    let spentDeep = false;
    try {
      const thread = await ownThread(req.params.threadId, userId);
      if (!thread) return fail(res, 404, `no thread ${req.params.threadId}`, requestId);
      if (ask.spaceId && !(await ownSpace(ask.spaceId, userId))) return fail(res, 404, `no space ${ask.spaceId}`, requestId);

      // The cap is checked before anything is spent, and the 429 says when to come back.
      if (ask.depth === 'deep') {
        if (!(await takeDeep(userId))) {
          return res.status(429).json({
            error: `deep search limit reached: ${env.deepDailyCap} per day`,
            status: 429,
            requestId,
            resetsAt: resetsAt()
          });
        }
        spentDeep = true;
      }

      const messages = (await db()).collection<MessageDoc>(COLLECTIONS.messages);
      const history = (
        await messages.find({ threadId: thread._id }).sort({ createdAt: -1 }).limit(HISTORY_MESSAGES).toArray()
      )
        .reverse()
        .map((m) => ({ role: m.role, content: m.content }));

      const run = ask.depth === 'deep' ? runDeep : runQuick;
      const result = await run({
        userId,
        threadId: thread._id,
        query: ask.query,
        mode: ask.mode,
        ...(ask.spaceId ? { spaceId: ask.spaceId } : {}),
        history,
        emit,
        state
      });

      const now = Date.now();
      await messages.insertMany([
        {
          _id: `msg_${randomUUID()}`,
          threadId: thread._id,
          userId,
          role: 'user',
          content: ask.query,
          sources: [],
          createdAt: new Date(now - 1)
        },
        {
          _id: `msg_${randomUUID()}`,
          threadId: thread._id,
          userId,
          role: 'assistant',
          content: result.text,
          answerId: result.done.answerId,
          sources: result.sources,
          done: result.done,
          ...(result.plan ? { subQuestions: result.plan } : {}),
          createdAt: new Date(now)
        }
      ]);
      if (thread.title === 'New thread') {
        await (await db())
          .collection<ThreadDoc>(COLLECTIONS.threads)
          .updateOne({ _id: thread._id }, { $set: { title: ask.query.slice(0, 80) } });
      }

      await writeRun(result.runLog, { answerId: result.done.answerId });
      emit('done', result.done);
      res.end();

      const d = result.done;
      log.info(
        {
          requestId,
          userId,
          toolCalls: result.runLog.toolCalls.length,
          terminated: d.terminated,
          depth: d.depth,
          tokens: d.tokens,
          costUsd: d.costUsd,
          searchCached: d.searchCached,
          ttftMs: d.ttftMs,
          latencyMs: d.latencyMs
        },
        'answer'
      );
    } catch (err) {
      const status = err instanceof ProviderError ? err.status : 502;
      const error = (err as Error).message || 'agent failed';
      const runLog: RunLog = {
        tokens: state.usage.in + state.usage.out,
        wallClockSec: Number(((Date.now() - state.started) / 1000).toFixed(3)),
        costUsd: Number(runCostUsd(state, 0).toFixed(6)),
        terminated: 'error',
        depth: ask.depth,
        toolCalls: state.toolCalls
      };
      log.error({ requestId, userId, err, terminated: 'error', depth: ask.depth, toolCalls: state.toolCalls.length }, 'answer failed');
      if (spentDeep) await refundDeep(userId).catch((e) => log.error({ requestId, err: e }, 'deep refund failed'));
      await writeRun(runLog, {}).catch((e) => log.error({ requestId, err: e }, 'run log write failed'));
      // Before the stream opened, the status line can still say 502. After, the error event does.
      if (!streaming) return fail(res, status, error, requestId);
      sseSend(res, 'error', { status, error });
      res.end();
    }
  });
}
