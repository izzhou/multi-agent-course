/**
 * Parse an upload into citable chunks. Every chunk carries a locator: {page} for a PDF,
 * {heading} (and {line}) for Markdown, {line} for plain text. Chunk text is verbatim
 * source text, because a doc citation's snippet is checked against the chunk itself.
 */
import type { Locator } from '@lumina/contract';

export interface ParsedChunk {
  text: string;
  locator: Locator;
  ord: number;
}

export interface Parsed {
  pages?: number;
  chunks: ParsedChunk[];
}

const TARGET_CHARS = 900;
const OVERLAP_CHARS = 150;

const squash = (s: string) => s.replace(/\s+/g, ' ').trim();

/** Split one block of text into windows of whole sentences, about TARGET_CHARS long, overlapping a little. */
export function window(text: string): string[] {
  // Split after . ! ? only when whitespace follows, so "1.2" and "e.g" stay whole.
  const sentences = text.split(/(?<=[.!?])\s+/).map((s) => s.trim()).filter(Boolean);
  const out: string[] = [];
  let cur: string[] = [];
  let len = 0;
  for (const s of sentences) {
    if (len + s.length > TARGET_CHARS && cur.length) {
      out.push(cur.join(' '));
      // carry the tail sentences forward as overlap
      const keep: string[] = [];
      let k = 0;
      for (let i = cur.length - 1; i >= 0 && k < OVERLAP_CHARS; i--) {
        keep.unshift(cur[i]!);
        k += cur[i]!.length;
      }
      cur = keep.length < cur.length ? keep : [];
      len = cur.join(' ').length;
    }
    cur.push(s);
    len += s.length + 1;
  }
  if (cur.length) out.push(cur.join(' '));
  return out;
}

async function parsePdf(buf: Buffer): Promise<Parsed> {
  const { getDocument } = await import('pdfjs-dist/legacy/build/pdf.mjs');
  const pdf = await getDocument({ data: new Uint8Array(buf), useSystemFonts: true, isEvalSupported: false }).promise;
  const chunks: ParsedChunk[] = [];
  for (let p = 1; p <= pdf.numPages; p++) {
    const page = await pdf.getPage(p);
    const content = await page.getTextContent();
    const text = squash(
      content.items.map((it) => ('str' in it ? it.str + (it.hasEOL ? '\n' : ' ') : '')).join('')
    );
    for (const w of window(text)) chunks.push({ text: w, locator: { page: p }, ord: chunks.length });
    page.cleanup();
  }
  const pages = pdf.numPages;
  await pdf.destroy();
  return { pages, chunks };
}

function parseMarkdown(src: string): Parsed {
  const chunks: ParsedChunk[] = [];
  const lines = src.split(/\r?\n/);
  let heading: string | undefined;
  let startLine = 1;
  let buf: string[] = [];
  const flush = () => {
    const text = squash(buf.join('\n'));
    if (text) {
      for (const w of window(text)) {
        chunks.push({ text: w, locator: { ...(heading ? { heading } : {}), line: startLine }, ord: chunks.length });
      }
    }
    buf = [];
  };
  lines.forEach((line, i) => {
    const h = line.match(/^#{1,6}\s+(.*)$/);
    if (h) {
      flush();
      heading = h[1]!.trim();
      startLine = i + 1;
    } else {
      if (!buf.length) startLine = i + 1;
      buf.push(line);
    }
  });
  flush();
  return { chunks };
}

function parseText(src: string): Parsed {
  const chunks: ParsedChunk[] = [];
  const paras = src.split(/\n\s*\n/);
  let line = 1;
  for (const para of paras) {
    const text = squash(para);
    if (text) for (const w of window(text)) chunks.push({ text: w, locator: { line }, ord: chunks.length });
    line += para.split('\n').length + 1;
  }
  return { chunks };
}

export async function parseDocument(buf: Buffer, mimeType: string, filename: string): Promise<Parsed> {
  const lower = filename.toLowerCase();
  if (mimeType === 'application/pdf' || lower.endsWith('.pdf')) return parsePdf(buf);
  if (mimeType === 'text/markdown' || lower.endsWith('.md')) return parseMarkdown(buf.toString('utf8'));
  return parseText(buf.toString('utf8'));
}
