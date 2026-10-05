/**
 * Long-term memory: stable facts and preferences, written only by an explicit save_memory
 * call, recalled semantically per user, listed at GET /memory and deletable. Nothing is
 * remembered that GET /memory does not show, because every read here goes to the same
 * collection the list reads.
 */
import { COLLECTIONS, SEARCH_INDEXES, newId, type MemoryDoc } from '@lumina/contract';
import { env } from './env.js';
import { db } from './db.js';
import { cosine, embed } from './embeddings.js';

export interface Recalled {
  id: string;
  text: string;
  score: number;
}

/** Atlas indexes are eventually consistent: a memory saved seconds ago may not be in the vector index yet. */
const RECENT_MS = 10 * 60 * 1000;

const memories = async () => (await db()).collection<MemoryDoc>(COLLECTIONS.memories);

export async function saveMemory(
  userId: string,
  text: string,
  sourceThread: string
): Promise<{ id: string; duplicate: boolean; embedTokens: number }> {
  const col = await memories();
  const existing = await col.findOne({ userId, text });
  if (existing) return { id: existing._id, duplicate: true, embedTokens: 0 };
  const { vectors, tokens } = await embed([text]);
  const doc: MemoryDoc = {
    _id: newId('mem'),
    userId,
    text,
    embedding: vectors[0]!,
    sourceThread: sourceThread as MemoryDoc['sourceThread'],
    createdAt: new Date()
  };
  await col.insertOne(doc);
  return { id: doc._id, duplicate: false, embedTokens: tokens };
}

/**
 * The user's memories most relevant to `query`, at most memoryRecallLimit. Semantic matches
 * come first; memories written in the last few minutes are added because the vector index
 * may not have caught up with them yet.
 */
export async function recallMemories(userId: string, query: string): Promise<{ memories: Recalled[]; embedTokens: number }> {
  const col = await memories();
  if ((await col.countDocuments({ userId }, { limit: 1 })) === 0) return { memories: [], embedTokens: 0 };

  const limit = env.memoryRecallLimit;
  const { vectors, tokens } = await embed([query]);
  const queryVector = vectors[0]!;

  let semantic: Recalled[];
  if (env.vectorBackend === 'atlas-vector-search') {
    semantic = await col
      .aggregate<Recalled>([
        {
          $vectorSearch: {
            index: SEARCH_INDEXES.memoriesVector,
            path: 'embedding',
            queryVector,
            numCandidates: limit * 10,
            limit,
            filter: { userId }
          }
        },
        { $project: { _id: 0, id: '$_id', text: 1, score: { $meta: 'vectorSearchScore' } } }
      ])
      .toArray();
  } else {
    const rows = await col.find({ userId }).project<Pick<MemoryDoc, '_id' | 'text' | 'embedding'>>({ text: 1, embedding: 1 }).toArray();
    semantic = rows
      .map((m) => ({ id: m._id, text: m.text, score: cosine(queryVector, m.embedding) }))
      .sort((a, b) => b.score - a.score)
      .slice(0, limit);
  }

  const recent = await col
    .find({ userId, createdAt: { $gte: new Date(Date.now() - RECENT_MS) } })
    .sort({ createdAt: -1 })
    .limit(limit)
    .project<Pick<MemoryDoc, '_id' | 'text'>>({ text: 1 })
    .toArray();

  const out = [...semantic];
  for (const m of recent) {
    if (!out.some((r) => r.id === m._id)) out.push({ id: m._id, text: m.text, score: 0 });
  }
  return { memories: out.slice(0, limit), embedTokens: tokens };
}

export async function listMemories(userId: string) {
  const rows = await (await memories())
    .find({ userId })
    .sort({ createdAt: -1 })
    .project<Omit<MemoryDoc, 'embedding'>>({ embedding: 0 })
    .toArray();
  return rows.map((m) => ({
    id: m._id,
    text: m.text,
    ...(m.sourceThread ? { sourceThread: m.sourceThread } : {}),
    createdAt: new Date(m.createdAt).toISOString()
  }));
}

export async function deleteMemory(userId: string, id: string): Promise<boolean> {
  const res = await (await memories()).deleteOne({ _id: id, userId });
  return res.deletedCount === 1;
}
