import Anthropic from '@anthropic-ai/sdk';
import { env, secrets } from './env.js';

/** One client per process. The key is read here and never leaves this service. */
export const anthropic = new Anthropic({ apiKey: secrets.anthropic || undefined, maxRetries: 1 });

export interface Usage {
  in: number;
  out: number;
}

export const addUsage = (total: Usage, u: { input_tokens: number; output_tokens: number }) => {
  total.in += u.input_tokens;
  total.out += u.output_tokens;
};

export const llmCostUsd = (u: Usage) =>
  (u.in * env.llmInputUsdPerMtok + u.out * env.llmOutputUsdPerMtok) / 1_000_000;

/** A provider failure. The loop ends the run with terminated:"error" and the caller sees a 502. */
export class ProviderError extends Error {
  constructor(message: string, readonly status = 502) {
    super(message);
  }
}

/** Shared request settings, so every call is priced and tuned the same way. */
export const baseParams = {
  model: env.llmModel,
  output_config: { effort: env.llmEffort }
} as const;
