/**
 * fetch_page: download a URL and keep the readable text (Readability over jsdom).
 * The answer is synthesised from THIS text, never from search snippets, and every cited
 * snippet is a verbatim passage of it, which is what the bench's grounding check looks for.
 */
import { JSDOM, VirtualConsole } from 'jsdom';
import { Readability } from '@mozilla/readability';

export interface Page {
  url: string;
  title: string;
  text: string;
}

const MAX_HTML_BYTES = 2_000_000;
const PAGE_TTL_MS = 10 * 60 * 1000;

/** In-flight and finished fetches, so a prefetch started by web_search is reused. */
const pages = new Map<string, { at: number; page: Promise<Page> }>();

const squash = (s: string) => s.replace(/\s+/g, ' ').trim();

async function download(url: string): Promise<Page> {
  const res = await fetch(url, {
    redirect: 'follow',
    headers: {
      'user-agent': 'Mozilla/5.0 (compatible; LuminaBot/0.1; course project)',
      accept: 'text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.5'
    },
    signal: AbortSignal.timeout(8000)
  });
  if (!res.ok) throw new Error(`fetch ${res.status} ${res.statusText}`.trim());
  const type = res.headers.get('content-type') ?? '';
  const raw = (await res.text()).slice(0, MAX_HTML_BYTES);

  if (type.includes('text/plain') || type.includes('markdown')) {
    return { url, title: url, text: squash(raw) };
  }
  if (!type.includes('html')) throw new Error(`unsupported content-type ${type || 'unknown'}`);

  const dom = new JSDOM(raw, { url, virtualConsole: new VirtualConsole() });
  const article = new Readability(dom.window.document).parse();
  const text = squash(article?.textContent ?? dom.window.document.body?.textContent ?? '');
  const title = squash(article?.title || dom.window.document.title || url);
  dom.window.close();
  if (text.length < 200) throw new Error('page has no readable text');
  return { url, title, text };
}

export function fetchPage(url: string): Promise<Page> {
  const hit = pages.get(url);
  if (hit && Date.now() - hit.at < PAGE_TTL_MS) return hit.page;
  const page = download(url);
  pages.set(url, { at: Date.now(), page });
  page.catch(() => pages.delete(url)); // a failure is not cached
  if (pages.size > 300) pages.delete(pages.keys().next().value!);
  return page;
}

/** Start fetching likely pages while the model decides which to read. Errors surface later, if it asks. */
export function prefetch(urls: string[]): void {
  for (const url of urls) fetchPage(url).catch(() => undefined);
}

// ---------------------------------------------------------------- passages

const STOP = new Set(
  'the a an and or of to in on for is are was were be by with what which who how why when does do did it its this that from as at'.split(
    ' '
  )
);

const terms = (s: string) =>
  s
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((t) => t.length > 1 && !STOP.has(t));

/**
 * The passage of `text` that best matches `query`: a verbatim window of whole sentences,
 * about 40 to 80 words. Verbatim matters: the snippet is checked against the page itself.
 */
export function bestPassage(text: string, query: string): string {
  // Split after . ! ? only when whitespace follows, so "1.2" and "e.g" stay whole.
  const sentences = text.split(/(?<=[.!?])\s+/).map((s) => s.trim()).filter(Boolean);
  const want = new Set(terms(query));
  let best = '';
  let bestScore = -1;
  for (let i = 0; i < sentences.length; i++) {
    let window = '';
    for (let j = i; j < sentences.length && window.split(' ').length < 40; j++) {
      window = window ? `${window} ${sentences[j]}` : sentences[j]!;
    }
    const words = terms(window);
    const score = words.filter((w) => want.has(w)).length / Math.sqrt(words.length + 1);
    if (score > bestScore) {
      bestScore = score;
      best = window;
    }
  }
  const words = best.split(' ');
  return words.length > 80 ? words.slice(0, 80).join(' ') : best;
}

/**
 * Up to `maxChars` of `text`, centred on the passage that best matches `query`. A long page
 * cut from the top often loses the one paragraph that answers the question, and deep search
 * reads many pages, so each one gets a smaller window in the right place.
 */
export function excerptAround(text: string, query: string, maxChars: number): string {
  if (text.length <= maxChars) return text;
  const at = Math.max(0, text.indexOf(bestPassage(text, query)));
  const start = Math.max(0, Math.min(at - Math.floor(maxChars / 3), text.length - maxChars));
  return `${start > 0 ? '… ' : ''}${text.slice(start, start + maxChars)}${start + maxChars < text.length ? ' …' : ''}`;
}
