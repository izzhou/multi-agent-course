/**
 * The jobs worker. One job kind, index_document:
 *
 *   GridFS read → parse (page-aware) → chunk → embed → upsert into chunks
 *   → READ-YOUR-WRITE PROBE → status: 'indexed'
 *
 * It runs off the answer path: in a worker_thread started by index.ts (the default), or as
 * its own process with `npm run worker`. Either way, parsing a 60-page PDF never blocks the
 * event loop that is streaming somebody's answer.
 *
 * Claims are atomic (findOneAndUpdate), so two workers never take the same job. A worker
 * killed mid-job leaves the row `running` with a stale claimedAt, and the sweeper returns
 * it to `pending`. Finished stages are not re-run: chunk ids are deterministic
 * (`<docId>:<ord>`), so a retried job embeds only the chunks it had not stored yet.
 */
import pino from 'pino';
import { randomUUID } from 'node:crypto';
import { GridFSBucket, ObjectId } from 'mongodb';
import {
  COLLECTIONS,
  GRIDFS_BUCKETS,
  type ChunkDoc,
  type DocumentDoc,
  type JobDoc
} from '@lumina/contract';
import { env } from './env.js';
import { db } from './db.js';
import { embed } from './embeddings.js';
import { parseDocument } from './parse.js';
import { probe } from './rag.js';

const log = pino({ level: env.logLevel, base: { svc: 'worker' } });
const workerId = `wrk_${randomUUID().slice(0, 8)}`;

const POLL_MS = 1000;
const STALE_MS = 2 * 60 * 1000;
const MAX_ATTEMPTS = 3;
const EMBED_BATCH = 64;
const PROBE_TIMEOUT_MS = 120_000;

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

async function setDoc(docId: string, set: Partial<DocumentDoc>): Promise<void> {
  await (await db()).collection<DocumentDoc>(COLLECTIONS.documents).updateOne({ _id: docId }, { $set: set });
}

async function readUpload(fileId: string): Promise<Buffer> {
  const bucket = new GridFSBucket(await db(), { bucketName: GRIDFS_BUCKETS.uploads });
  const parts: Buffer[] = [];
  for await (const part of bucket.openDownloadStream(new ObjectId(fileId))) parts.push(part as Buffer);
  return Buffer.concat(parts);
}

async function indexDocument(job: JobDoc): Promise<void> {
  const database = await db();
  const docId = String(job.payload.docId);
  const doc = await database.collection<DocumentDoc>(COLLECTIONS.documents).findOne({ _id: docId });
  if (!doc) throw new Error(`document ${docId} no longer exists`);

  // parse + chunk
  await setDoc(docId, { status: 'parsing', pct: 10 });
  const parsed = await parseDocument(await readUpload(doc.fileId), doc.mimeType, doc.title);
  if (!parsed.chunks.length) throw new Error('no extractable text (a scanned PDF needs OCR)');
  await setDoc(docId, { status: 'embedding', pct: 30, ...(parsed.pages ? { pages: parsed.pages } : {}) });

  // embed + upsert, skipping chunks a previous attempt already stored
  const chunksCol = database.collection<ChunkDoc>(COLLECTIONS.chunks);
  const ids = parsed.chunks.map((c) => `${docId}:${c.ord}`);
  const done = new Set((await chunksCol.find({ _id: { $in: ids } }).project({ _id: 1 }).toArray()).map((r) => String(r._id)));
  const todo = parsed.chunks.filter((c) => !done.has(`${docId}:${c.ord}`));

  for (let i = 0; i < todo.length; i += EMBED_BATCH) {
    const batch = todo.slice(i, i + EMBED_BATCH);
    const { vectors } = await embed(batch.map((c) => c.text));
    await chunksCol.bulkWrite(
      batch.map((c, j) => ({
        replaceOne: {
          filter: { _id: `${docId}:${c.ord}` },
          replacement: {
            _id: `${docId}:${c.ord}`,
            docId: doc._id,
            spaceId: doc.spaceId,
            userId: doc.userId,
            text: c.text,
            locator: c.locator,
            ord: c.ord,
            embedding: vectors[j]!,
            createdAt: new Date()
          },
          upsert: true
        }
      }))
    );
    await setDoc(docId, { pct: Math.round(30 + (60 * (i + batch.length)) / todo.length) });
  }

  // read-your-write probe: "upserted" is not "searchable"
  const first = await chunksCol.findOne({ _id: ids[0]! });
  if (!first) throw new Error('chunk vanished before the probe');
  const deadline = Date.now() + PROBE_TIMEOUT_MS;
  for (let wait = 500; ; wait = Math.min(wait * 2, 8000)) {
    if (await probe(first)) break;
    if (Date.now() > deadline) throw new Error(`vector index did not return the document within ${PROBE_TIMEOUT_MS / 1000}s`);
    await sleep(wait);
  }

  await setDoc(docId, { status: 'indexed', pct: 100, chunks: parsed.chunks.length });
  log.info({ docId, chunks: parsed.chunks.length, pages: parsed.pages }, 'document indexed');
}

/** Return crashed jobs to the queue. A job that keeps crashing is failed, not retried forever. */
async function sweep(): Promise<void> {
  const jobs = (await db()).collection<JobDoc>(COLLECTIONS.jobs);
  const stale = { status: 'running' as const, claimedAt: { $lt: new Date(Date.now() - STALE_MS) } };
  await jobs.updateMany({ ...stale, attempts: { $lt: MAX_ATTEMPTS } }, { $set: { status: 'pending' }, $unset: { workerId: '' } });
  const dead = await jobs.find({ ...stale, attempts: { $gte: MAX_ATTEMPTS } }).toArray();
  for (const job of dead) {
    await jobs.updateOne({ _id: job._id }, { $set: { status: 'failed', error: 'worker crashed on every attempt' } });
    await setDoc(String(job.payload.docId), { status: 'failed', error: 'worker crashed on every attempt' });
  }
}

async function claim(): Promise<JobDoc | null> {
  return (await db()).collection<JobDoc>(COLLECTIONS.jobs).findOneAndUpdate(
    { status: 'pending' },
    { $set: { status: 'running', claimedAt: new Date(), workerId }, $inc: { attempts: 1 } },
    { sort: { createdAt: 1 }, returnDocument: 'after' }
  );
}

async function runOne(job: JobDoc): Promise<void> {
  const jobs = (await db()).collection<JobDoc>(COLLECTIONS.jobs);
  try {
    await indexDocument(job);
    await jobs.updateOne({ _id: job._id }, { $set: { status: 'done' } });
  } catch (err) {
    // Fail loud: the error string reaches GET /spaces/:id/documents, never a fake "indexed".
    const error = (err as Error).message || 'indexing failed';
    log.error({ jobId: job._id, docId: job.payload.docId, err }, 'job failed');
    await jobs.updateOne({ _id: job._id }, { $set: { status: 'failed', error } });
    await setDoc(String(job.payload.docId), { status: 'failed', error });
  }
}

async function main(): Promise<void> {
  log.info({ workerId }, 'jobs worker up');
  let lastSweep = 0;
  for (;;) {
    try {
      if (Date.now() - lastSweep > 30_000) {
        await sweep();
        lastSweep = Date.now();
      }
      const job = await claim();
      if (job) await runOne(job);
      else await sleep(POLL_MS);
    } catch (err) {
      // Mongo unreachable: keep the worker alive and say so, then try again.
      log.error({ err }, 'worker loop error');
      await sleep(5000);
    }
  }
}

void main();
