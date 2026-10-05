/**
 * The gateway's half of every contract route: check X-User-Id, validate the body with the
 * contract's zod schema, apply the per-user rate limit, then forward to the agent service
 * and stream the answer back unbuffered. An unreachable agent is a 502, never a 2xx.
 */
import type express from 'express';
import { Readable } from 'node:stream';
import type { ReadableStream as WebReadableStream } from 'node:stream/web';
import type { ZodTypeAny } from 'zod';
import { AskBody, CreateSpaceBody, CreateThreadBody, REQUEST_HEADER, USER_HEADER } from '@lumina/contract';
import { env } from './env.js';
import { sseHeaders } from './sse.js';

type Handler = (req: express.Request, res: express.Response) => Promise<void> | void;

const fail = (res: express.Response, status: number, error: string, extra: Record<string, unknown> = {}) => {
  res.status(status).json({ error, status, requestId: String(res.locals.requestId), ...extra });
};

/** Request bodies the contract defines, keyed by "METHOD path". */
const BODIES: Record<string, ZodTypeAny> = {
  'POST /threads': CreateThreadBody,
  'POST /threads/:threadId/ask': AskBody,
  'POST /spaces': CreateSpaceBody
};

// ---------------------------------------------------------------- rate limit

/** A fixed one-minute window per user. Per process: fine for one instance, not for a fleet. */
const windows = new Map<string, { start: number; count: number }>();

function overLimit(userId: string): { resetsAt: string } | null {
  const now = Date.now();
  const w = windows.get(userId);
  if (!w || now - w.start >= 60_000) {
    windows.set(userId, { start: now, count: 1 });
    return null;
  }
  w.count++;
  return w.count > env.rateLimitPerMinute ? { resetsAt: new Date(w.start + 60_000).toISOString() } : null;
}

// ---------------------------------------------------------------- the proxy

export function proxy(routeKey: string): Handler {
  const schema = BODIES[routeKey];
  const isUpload = routeKey === 'POST /spaces/:spaceId/documents';

  return async (req, res) => {
    const userId = req.header(USER_HEADER)?.trim();
    if (!userId) return fail(res, 401, 'X-User-Id header is required');

    if (schema) {
      const parsed = schema.safeParse(req.body ?? {});
      if (!parsed.success) return fail(res, 400, parsed.error.issues[0]?.message ?? 'invalid body');
    }

    const limited = overLimit(userId);
    if (limited) return fail(res, 429, `rate limit: ${env.rateLimitPerMinute} requests per minute`, limited);

    const headers: Record<string, string> = {
      [USER_HEADER]: userId,
      [REQUEST_HEADER]: String(res.locals.requestId)
    };
    let body: RequestInit["body"];
    if (isUpload) {
      // Multipart goes through as a stream: the gateway never buffers an upload.
      for (const h of ['content-type', 'content-length']) {
        const v = req.header(h);
        if (v) headers[h] = v;
      }
      body = Readable.toWeb(req) as unknown as RequestInit["body"];
    } else if (req.method !== 'GET' && req.method !== 'DELETE') {
      headers['content-type'] = 'application/json';
      body = JSON.stringify(req.body ?? {});
    }

    const abort = new AbortController();
    res.on('close', () => abort.abort());

    let upstream: Response;
    try {
      upstream = await fetch(`${env.agentUrl}${req.originalUrl}`, {
        method: req.method,
        headers,
        body,
        signal: abort.signal,
        // Required by Node's fetch for a streamed request body.
        ...(isUpload ? { duplex: 'half' } : {})
      } as RequestInit);
    } catch (err) {
      return fail(res, 502, `agent service unreachable: ${(err as Error).message}`);
    }

    const type = upstream.headers.get('content-type') ?? '';
    if (type.includes('text/event-stream') && upstream.body) {
      res.status(upstream.status);
      sseHeaders(res);
      // Pass each chunk through as it arrives. Buffering here is what turns a stream into a wait.
      try {
        for await (const chunk of Readable.fromWeb(upstream.body as unknown as WebReadableStream)) {
          res.write(chunk);
        }
      } catch (err) {
        if (!abort.signal.aborted) {
          res.write(`event: error\ndata: ${JSON.stringify({ status: 502, error: `agent stream broke: ${(err as Error).message}` })}\n\n`);
        }
      }
      res.end();
      return;
    }

    res.status(upstream.status);
    if (type) res.setHeader('content-type', type);
    res.send(Buffer.from(await upstream.arrayBuffer()));
  };
}
