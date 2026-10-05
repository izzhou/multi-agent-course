/**
 * Retrieval over a Space's documents: dense ($vectorSearch) and lexical ($search, BM25)
 * fused with reciprocal rank fusion. spaceId sits INSIDE both queries as a filter, so
 * another Space's chunks never take a top-k slot.
 *
 * No re-rank step: the corpus is small and RRF over two retrievers already meets the
 * recall target, and a cross-encoder call would add latency to every docs answer. If
 * recall@5 slips, a re-ranker is the first thing to add.
 */
import { COLLECTIONS, SEARCH_INDEXES, type ChunkDoc, type DocumentDoc, type Locator } from '@lumina/contract';
import { env } from './env.js';
import { db } from './db.js';
import { cosine, embed } from './embeddings.js';

export interface DocHit {
  chunkId: string;
  docId: string;
  title: string;
  text: string;
  locator: Locator;
  score: number;
}

type Row = Pick<ChunkDoc, '_id' | 'docId' | 'text' | 'locator'>;

const chunks = async () => (await db()).collection<ChunkDoc>(COLLECTIONS.chunks);

async function denseAtlas(spaceId: string, userId: string, vector: number[]): Promise<Row[]> {
  return (await chunks())
    .aggregate<Row>([
      {
        $vectorSearch: {
          index: SEARCH_INDEXES.chunksVector,
          path: 'embedding',
          queryVector: vector,
          numCandidates: env.docsCandidates * 10,
          limit: env.docsCandidates,
          filter: { spaceId, userId }
        }
      },
      { $project: { text: 1, docId: 1, locator: 1 } }
    ])
    .toArray();
}

async function lexicalAtlas(spaceId: string, userId: string, query: string): Promise<Row[]> {
  return (await chunks())
    .aggregate<Row>([
      {
        $search: {
          index: SEARCH_INDEXES.chunksText,
          compound: {
            must: [{ text: { query, path: 'text' } }],
            filter: [{ equals: { path: 'spaceId', value: spaceId } }, { equals: { path: 'userId', value: userId } }]
          }
        }
      },
      { $limit: env.docsCandidates },
      { $project: { text: 1, docId: 1, locator: 1 } }
    ])
    .toArray();
}

const words = (s: string) => s.toLowerCase().match(/[a-z0-9]+/g) ?? [];

/** The local-dev fallback: exact cosine and a crude term-frequency score, in Node. Fine to a few thousand chunks. */
async function scan(spaceId: string, userId: string, query: string, vector: number[]): Promise<[Row[], Row[]]> {
  const rows = await (await chunks()).find({ spaceId, userId }).project<Row & { embedding: number[] }>({ text: 1, docId: 1, locator: 1, embedding: 1 }).toArray();
  const dense = [...rows].sort((a, b) => cosine(vector, b.embedding) - cosine(vector, a.embedding)).slice(0, env.docsCandidates);
  const q = new Set(words(query).filter((w) => w.length > 2));
  const tf = (t: string) => words(t).filter((w) => q.has(w)).length;
  const lexical = rows.map((r) => ({ r, s: tf(r.text) })).filter((x) => x.s > 0).sort((a, b) => b.s - a.s).slice(0, env.docsCandidates).map((x) => x.r);
  return [dense, lexical];
}

export async function searchDocuments(
  userId: string,
  spaceId: string,
  query: string
): Promise<{ hits: DocHit[]; embedTokens: number }> {
  const { vectors, tokens } = await embed([query]);
  const vector = vectors[0]!;

  const [dense, lexical] =
    env.vectorBackend === 'atlas-vector-search'
      ? await Promise.all([denseAtlas(spaceId, userId, vector), lexicalAtlas(spaceId, userId, query)])
      : await scan(spaceId, userId, query, vector);

  // Reciprocal rank fusion: a chunk's score is the sum of 1/(k + rank) over both lists.
  const fused = new Map<string, { row: Row; score: number }>();
  for (const list of [dense, lexical]) {
    list.forEach((row, rank) => {
      const cur = fused.get(row._id) ?? { row, score: 0 };
      cur.score += 1 / (env.rrfK + rank + 1);
      fused.set(row._id, cur);
    });
  }
  const top = [...fused.values()].sort((a, b) => b.score - a.score).slice(0, env.docsTopK);

  const docIds = [...new Set(top.map((t) => t.row.docId))];
  const titles = new Map(
    (await (await db()).collection<DocumentDoc>(COLLECTIONS.documents).find({ _id: { $in: docIds } }).project<Pick<DocumentDoc, '_id' | 'title'>>({ title: 1 }).toArray()).map((d) => [d._id, d.title])
  );

  return {
    hits: top.map(({ row, score }) => ({
      chunkId: row._id,
      docId: row.docId,
      title: titles.get(row.docId) ?? row.docId,
      text: row.text,
      locator: row.locator,
      score
    })),
    embedTokens: tokens
  };
}

/** Does the Space hold anything searchable? The router asks before it spends a search. */
export async function spaceHasIndexedDocs(userId: string, spaceId: string): Promise<boolean> {
  const n = await (await db()).collection<DocumentDoc>(COLLECTIONS.documents).countDocuments({ spaceId, userId, status: 'indexed' }, { limit: 1 });
  return n > 0;
}

/**
 * The read-your-write probe: query the vector index with one chunk's own embedding and
 * require that chunk's document back. Only then is the document `indexed`.
 */
export async function probe(chunk: Pick<ChunkDoc, 'docId' | 'spaceId' | 'userId' | 'embedding'>): Promise<boolean> {
  if (env.vectorBackend !== 'atlas-vector-search') {
    return (await (await chunks()).countDocuments({ docId: chunk.docId }, { limit: 1 })) > 0;
  }
  const rows = await denseAtlas(chunk.spaceId, chunk.userId, chunk.embedding);
  return rows.some((r) => r.docId === chunk.docId);
}
