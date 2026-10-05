/**
 * The per-user daily deep search cap. One counter document per user per UTC day,
 * incremented atomically, so two deep asks racing at the cap cannot both get through.
 */
import { MongoServerError } from 'mongodb';
import { env } from './env.js';
import { db } from './db.js';

/** Agent-owned bookkeeping, not part of the shared contract's collection list. */
const COLLECTION = 'deepCounters';

interface CounterDoc {
  _id: string;
  userId: string;
  day: string;
  count: number;
  expiresAt: Date;
}

const today = () => new Date().toISOString().slice(0, 10);

/** The next UTC midnight: when today's counter stops counting. */
export const resetsAt = () => {
  const d = new Date();
  d.setUTCHours(24, 0, 0, 0);
  return d.toISOString();
};

let indexed = false;
async function counters() {
  const col = (await db()).collection<CounterDoc>(COLLECTION);
  if (!indexed) {
    indexed = true;
    // Old days delete themselves. Created here because the index script is not ours to edit.
    await col.createIndex({ expiresAt: 1 }, { expireAfterSeconds: 0 }).catch(() => (indexed = false));
  }
  return col;
}

/** Spends one deep search for today, or returns false when the cap is already reached. */
export async function takeDeep(userId: string): Promise<boolean> {
  const day = today();
  const col = await counters();
  try {
    const doc = await col.findOneAndUpdate(
      { _id: `${userId}:${day}`, count: { $lt: env.deepDailyCap } },
      { $inc: { count: 1 }, $setOnInsert: { userId, day, expiresAt: new Date(Date.now() + 2 * 86_400_000) } },
      { upsert: true, returnDocument: 'after' }
    );
    return !!doc;
  } catch (err) {
    // The filter missed because count is at the cap, so the upsert hit the existing _id.
    if (err instanceof MongoServerError && err.code === 11000) return false;
    throw err;
  }
}

/** Gives one back: a run that failed on our side should not cost the user their quota. */
export async function refundDeep(userId: string): Promise<void> {
  await (await counters()).updateOne({ _id: `${userId}:${today()}`, count: { $gt: 0 } }, { $inc: { count: -1 } });
}

export async function deepToday(userId: string): Promise<number> {
  return (await (await counters()).findOne({ _id: `${userId}:${today()}` }))?.count ?? 0;
}
