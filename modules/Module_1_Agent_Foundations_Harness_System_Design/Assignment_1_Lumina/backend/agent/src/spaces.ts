/**
 * Spaces and document upload. The upload route does the minimum on the request path:
 * store the file in GridFS, insert a `pending` document and a `jobs` row, answer 202.
 * Everything slow happens on the jobs worker.
 */
import express from 'express';
import multer from 'multer';
import { GridFSBucket } from 'mongodb';
import {
  ACCEPTED_UPLOAD_TYPES,
  COLLECTIONS,
  CreateSpaceBody,
  GRIDFS_BUCKETS,
  MAX_UPLOAD_BYTES,
  SpaceId,
  newId,
  type DocumentDoc,
  type JobDoc,
  type ListDocumentsResponse,
  type ListSpacesResponse,
  type SpaceDoc
} from '@lumina/contract';
import { db } from './db.js';

const fail = (res: express.Response, status: number, error: string) => res.status(status).json({ error, status });

const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: MAX_UPLOAD_BYTES, files: 1 } });

const EXT_TYPES: Record<string, (typeof ACCEPTED_UPLOAD_TYPES)[number]> = {
  pdf: 'application/pdf',
  md: 'text/markdown',
  markdown: 'text/markdown',
  txt: 'text/plain'
};

/** The browser's guess at a type is unreliable for .md, so the extension decides when it can. */
function mimeOf(file: Express.Multer.File): string | null {
  const ext = file.originalname.split('.').pop()?.toLowerCase() ?? '';
  if (EXT_TYPES[ext]) return EXT_TYPES[ext]!;
  return (ACCEPTED_UPLOAD_TYPES as readonly string[]).includes(file.mimetype) ? file.mimetype : null;
}

export async function ownSpace(spaceId: string, userId: string) {
  if (!SpaceId.safeParse(spaceId).success) return null;
  return (await db()).collection<SpaceDoc>(COLLECTIONS.spaces).findOne({ _id: spaceId, userId });
}

export function registerSpaceRoutes(app: express.Express): void {
  app.post('/spaces', async (req, res, next) => {
    try {
      const body = CreateSpaceBody.safeParse(req.body ?? {});
      if (!body.success) return fail(res, 400, body.error.issues[0]?.message ?? 'invalid body');
      const doc: SpaceDoc = { _id: newId('spc'), userId: res.locals.userId, name: body.data.name, createdAt: new Date() };
      await (await db()).collection<SpaceDoc>(COLLECTIONS.spaces).insertOne(doc);
      res.status(201).json({ spaceId: doc._id, name: doc.name });
    } catch (err) {
      next(err);
    }
  });

  app.get('/spaces', async (_req, res, next) => {
    try {
      const rows = await (await db()).collection<SpaceDoc>(COLLECTIONS.spaces).find({ userId: res.locals.userId }).sort({ createdAt: -1 }).toArray();
      const body: ListSpacesResponse = {
        spaces: rows.map((s) => ({ spaceId: s._id, name: s.name, createdAt: new Date(s.createdAt).toISOString() }))
      };
      res.json(body);
    } catch (err) {
      next(err);
    }
  });

  app.post(
    '/spaces/:spaceId/documents',
    (req, res, next) =>
      upload.single('file')(req, res, (err: unknown) => {
        if (err instanceof multer.MulterError && err.code === 'LIMIT_FILE_SIZE') {
          return fail(res, 413, `file too large: the limit is ${MAX_UPLOAD_BYTES / 1024 / 1024} MB`);
        }
        if (err) return fail(res, 400, (err as Error).message);
        next();
      }),
    async (req, res, next) => {
      try {
        const userId: string = res.locals.userId;
        const space = await ownSpace(req.params.spaceId!, userId);
        if (!space) return fail(res, 404, `no space ${req.params.spaceId}`);
        const file = req.file;
        if (!file) return fail(res, 400, 'a multipart field named "file" is required');
        const mimeType = mimeOf(file);
        if (!mimeType) return fail(res, 400, `unsupported file type ${file.mimetype}: upload PDF, Markdown or plain text`);

        const database = await db();
        const docId = newId('doc');
        const bucket = new GridFSBucket(database, { bucketName: GRIDFS_BUCKETS.uploads });
        const stream = bucket.openUploadStream(file.originalname, { metadata: { docId, spaceId: space._id, userId, mimeType } });
        await new Promise<void>((resolve, reject) => {
          stream.once('finish', () => resolve());
          stream.once('error', reject);
          stream.end(file.buffer);
        });

        const now = new Date();
        await database.collection<DocumentDoc>(COLLECTIONS.documents).insertOne({
          _id: docId,
          spaceId: space._id,
          userId,
          title: file.originalname,
          mimeType,
          bytes: file.size,
          status: 'pending',
          pct: 0,
          fileId: stream.id.toString(),
          createdAt: now
        });
        await database.collection<JobDoc>(COLLECTIONS.jobs).insertOne({
          _id: `job_${docId}`,
          kind: 'index_document',
          status: 'pending',
          payload: { docId },
          userId,
          attempts: 0,
          createdAt: now
        });
        res.status(202).json({ docId, status: 'pending' });
      } catch (err) {
        next(err);
      }
    }
  );

  app.get('/spaces/:spaceId/documents', async (req, res, next) => {
    try {
      const space = await ownSpace(req.params.spaceId, res.locals.userId);
      if (!space) return fail(res, 404, `no space ${req.params.spaceId}`);
      const rows = await (await db()).collection<DocumentDoc>(COLLECTIONS.documents).find({ spaceId: space._id }).sort({ createdAt: 1 }).toArray();
      const body: ListDocumentsResponse = {
        documents: rows.map((d) => ({
          docId: d._id,
          title: d.title,
          status: d.status,
          pct: d.pct,
          ...(d.pages ? { pages: d.pages } : {}),
          ...(d.chunks !== undefined ? { chunks: d.chunks } : {}),
          ...(d.error ? { error: d.error } : {})
        }))
      };
      res.json(body);
    } catch (err) {
      next(err);
    }
  });
}
