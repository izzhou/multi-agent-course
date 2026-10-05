import OpenAI from 'openai';
import { EMBEDDING_DIMS } from '@lumina/contract';
import { env, secrets } from './env.js';
import { ProviderError } from './llm.js';

const openai = new OpenAI({ apiKey: secrets.openai || 'missing', maxRetries: 1 });

export interface Embedded {
  vectors: number[][];
  tokens: number;
}

/** Embed texts in one call. A failure throws a ProviderError: no zero vector stands in for a real one. */
export async function embed(texts: string[]): Promise<Embedded> {
  if (!secrets.openai) throw new ProviderError('OPENAI_API_KEY is not set (embeddings need it)');
  try {
    const res = await openai.embeddings.create({ model: env.embeddingModel, input: texts });
    const vectors = res.data.sort((a, b) => a.index - b.index).map((d) => d.embedding);
    if (vectors.some((v) => v.length !== EMBEDDING_DIMS)) {
      throw new Error(`expected ${EMBEDDING_DIMS} dimensions from ${env.embeddingModel}`);
    }
    return { vectors, tokens: res.usage.total_tokens };
  } catch (err) {
    throw new ProviderError(`embeddings: ${(err as Error).message}`);
  }
}

export const embeddingCostUsd = (tokens: number) => (tokens * env.embeddingUsdPerMtok) / 1_000_000;

export function cosine(a: number[], b: number[]): number {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i]! * b[i]!;
    na += a[i]! * a[i]!;
    nb += b[i]! * b[i]!;
  }
  return na && nb ? dot / Math.sqrt(na * nb) : 0;
}
