/**
 * web_search: one interface over Tavily and SerpApi, picked by SEARCH_PROVIDER, behind a
 * two tier cache (in-process LRU, then the searchCache collection with a TTL index).
 *
 * Errors are thrown, never swallowed: the loop turns a thrown search into a trace step with
 * ok:false, and an outage where every search failed into a 502.
 */
import { createHash } from 'node:crypto';
import { COLLECTIONS, type SearchCacheDoc } from '@lumina/contract';
import { env, secrets } from './env.js';
import { db } from './db.js';

export interface SearchResult {
  title: string;
  url: string;
  snippet: string;
}

export interface SearchOutcome {
  results: SearchResult[];
  cached: boolean;
}

// ---------------------------------------------------------------- the providers

async function tavily(query: string): Promise<SearchResult[]> {
  if (!secrets.tavily) throw new Error('TAVILY_API_KEY is not set');
  const res = await fetch('https://api.tavily.com/search', {
    method: 'POST',
    headers: { 'content-type': 'application/json', authorization: `Bearer ${secrets.tavily}` },
    body: JSON.stringify({ query, max_results: env.searchMaxResults, search_depth: 'basic' }),
    signal: AbortSignal.timeout(10000)
  });
  if (!res.ok) throw new Error(`tavily ${res.status}: ${(await res.text()).slice(0, 200)}`);
  const body = (await res.json()) as { results?: { title?: string; url?: string; content?: string }[] };
  return (body.results ?? [])
    .filter((r) => r.url)
    .map((r) => ({ title: r.title || r.url!, url: r.url!, snippet: r.content ?? '' }));
}

async function serpapi(query: string): Promise<SearchResult[]> {
  if (!secrets.serpapi) throw new Error('SERPAPI_API_KEY is not set');
  const url = new URL('https://serpapi.com/search.json');
  url.searchParams.set('engine', 'google');
  url.searchParams.set('q', query);
  url.searchParams.set('num', String(env.searchMaxResults));
  url.searchParams.set('api_key', secrets.serpapi);
  const res = await fetch(url, { signal: AbortSignal.timeout(10000) });
  if (!res.ok) throw new Error(`serpapi ${res.status}: ${(await res.text()).slice(0, 200)}`);
  const body = (await res.json()) as {
    error?: string;
    organic_results?: { title?: string; link?: string; snippet?: string }[];
  };
  if (body.error) throw new Error(`serpapi: ${body.error}`);
  return (body.organic_results ?? [])
    .filter((r) => r.link)
    .slice(0, env.searchMaxResults)
    .map((r) => ({ title: r.title || r.link!, url: r.link!, snippet: r.snippet ?? '' }));
}

const providers = { tavily, serpapi } as const;

// ---------------------------------------------------------------- the cache

export const normalizeQuery = (q: string) => q.toLowerCase().replace(/\s+/g, ' ').trim();

export const cacheKey = (query: string, provider: string) =>
  createHash('sha256').update(`${provider}\u0000${normalizeQuery(query)}`).digest('hex');

/** Fresh facts should not be served from a six hour old search. */
const TIME_SENSITIVE = new RegExp(
  `\\b(today|tonight|now|latest|breaking|this (week|morning)|${new Date().getFullYear()}|${new Date().getFullYear() + 1})\\b`,
  'i'
);

class Lru<V> {
  private map = new Map<string, { value: V; expiresAt: number }>();
  constructor(private readonly max: number) {}
  get(key: string): V | undefined {
    const hit = this.map.get(key);
    if (!hit) return undefined;
    this.map.delete(key);
    if (hit.expiresAt < Date.now()) return undefined;
    this.map.set(key, hit); // refresh recency
    return hit.value;
  }
  set(key: string, value: V, ttlMs: number): void {
    this.map.delete(key);
    this.map.set(key, { value, expiresAt: Date.now() + ttlMs });
    if (this.map.size > this.max) this.map.delete(this.map.keys().next().value!);
  }
}

const lru = new Lru<SearchResult[]>(500);

/** Counters for /stats, reset with the process. */
export const searchStats = { hits: 0, misses: 0 };

export async function webSearch(query: string): Promise<SearchOutcome> {
  const provider = env.searchProvider;
  const key = cacheKey(query, provider);
  const ttlMs = env.searchCacheTtlSeconds * 1000;
  const bypass = TIME_SENSITIVE.test(query);

  if (!bypass) {
    const mem = lru.get(key);
    if (mem) {
      searchStats.hits++;
      return { results: mem, cached: true };
    }
    const row = await (await db())
      .collection<SearchCacheDoc>(COLLECTIONS.searchCache)
      .findOne({ _id: key, expiresAt: { $gt: new Date() } });
    if (row) {
      const results = row.results as unknown as SearchResult[];
      lru.set(key, results, new Date(row.expiresAt).getTime() - Date.now());
      searchStats.hits++;
      return { results, cached: true };
    }
  }

  searchStats.misses++;
  const results = await providers[provider](query);
  lru.set(key, results, ttlMs);
  const now = new Date();
  await (await db())
    .collection<SearchCacheDoc>(COLLECTIONS.searchCache)
    .updateOne(
      { _id: key },
      {
        $set: {
          provider,
          query: normalizeQuery(query),
          results: results as unknown as Record<string, unknown>[],
          expiresAt: new Date(now.getTime() + ttlMs),
          createdAt: now
        }
      },
      { upsert: true }
    );
  return { results, cached: false };
}
