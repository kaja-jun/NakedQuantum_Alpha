/* WATCHER v2.0 -- NakedQuantum
   BGE-base-en-v1.5 via Transformers.js
   Separate IndexedDB nq_watcher
   Vanilla JS cosine on Float32Array
*/

importScripts('https://cdn.jsdelivr.net/npm/@xenova/transformers@2.15.0/dist/transformers.min.js');

const MODEL_ID = 'Xenova/bge-base-en-v1.5';
const SIMILARITY_THRESHOLD = 0.82;
const DECAY_ACTIVE_DAYS = 25;
const LINK_EXPIRY_DAYS = 60;
const SILENT_PERIOD_HOURS = 48;
const TITLE_WEIGHT = 0.3;
const BODY_WEIGHT = 0.7;
const SPARK_BOOST = 1.1;
const TEMPORAL_BOOST_MAX = 0.15;
const RECENT_DAYS = 7;
const DAY_MS = 86400000;

let embedder = null;
let watcherDB = null;
let isReady = false;
let lastPassTime = 0;

/* ── INDEXEDDB ─────────────────────────────────────────────── */

function openWatcherDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open('nq_watcher', 1);
    req.onupgradeneeded = (e) => {
      const db = e.target.result;
      if (!db.objectStoreNames.contains('embeddings')) {
        const emb = db.createObjectStore('embeddings', { keyPath: 'id' });
        emb.createIndex('last_seen_active', 'last_seen_active', { unique: false });
      }
      if (!db.objectStoreNames.contains('links')) {
        const lnk = db.createObjectStore('links', { keyPath: 'id' });
        lnk.createIndex('created_at', 'created_at', { unique: false });
        lnk.createIndex('a', 'a', { unique: false });
        lnk.createIndex('b', 'b', { unique: false });
      }
    };
    req.onsuccess = (e) => resolve(e.target.result);
    req.onerror = (e) => reject(e.target.error);
  });
}

const idb = {
  get: (store, key) => new Promise((res) => {
    const r = watcherDB.transaction(store, 'readonly').objectStore(store).get(key);
    r.onsuccess = () => res(r.result || null);
    r.onerror = () => res(null);
  }),
  put: (store, obj) => new Promise((res, rej) => {
    const r = watcherDB.transaction(store, 'readwrite').objectStore(store).put(obj);
    r.onsuccess = () => res();
    r.onerror = (e) => rej(e.target.error);
  }),
  delete: (store, key) => new Promise((res) => {
    const r = watcherDB.transaction(store, 'readwrite').objectStore(store).delete(key);
    r.onsuccess = () => res();
    r.onerror = () => res();
  }),
  getAll: (store) => new Promise((res) => {
    const r = watcherDB.transaction(store, 'readonly').objectStore(store).getAll();
    r.onsuccess = () => res(r.result || []);
    r.onerror = () => res([]);
  }),
  deleteOlderThan: (store, indexName, cutoff) => new Promise((res) => {
    const range = IDBKeyRange.upperBound(cutoff);
    const req = watcherDB.transaction(store, 'readwrite').objectStore(store).index(indexName).openCursor(range);
    req.onsuccess = (e) => {
      const cursor = e.target.result;
      if (cursor) { cursor.delete(); cursor.continue(); } else res();
    };
    req.onerror = () => res();
  })
};

/* ── SHA256 ────────────────────────────────────────────────── */

async function sha256(text) {
  const buf = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text));
  return Array.from(new Uint8Array(buf)).map(b => b.toString(16).padStart(2, '0')).join('');
}

/* ── COSINE ─────────────────────────────────────────────────── */

function cosine(a, b) {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot   += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  if (normA === 0 || normB === 0) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

/* ── LOAD MODEL ─────────────────────────────────────────────── */

async function loadModel() {
  try {
    const T = self.Transformers || self.transformers;
    self.postMessage({ type: 'WATCHER_STATUS', status: 'loading' });
    T.env.useBrowserCache = true;
    T.env.backends.onnx.wasm.proxy = true;
    T.env.backends.onnx.wasm.numThreads = 1;
    embedder = await T.pipeline(
      'feature-extraction', MODEL_ID, {
        quantized: true,
        progress_callback: (progress) => {
          if (progress.status === 'downloading') {
            self.postMessage({
              type: 'MODEL_PROGRESS',
              loaded: Math.round((progress.loaded || 0) / 1024 / 1024),
              total: Math.round((progress.total || 136 * 1024 * 1024) / 1024 / 1024)
            });
          }
        }
      }
    );
    return true;
  } catch (err) {
    self.postMessage({ type: 'WATCHER_STATUS', status: 'error', reason: err.message });
    return false;
  }
}

/* ── EMBED ──────────────────────────────────────────────────── */

async function embedDiscourse(id, title, body, itemType, lastSeenActive) {
  if (!embedder) throw new Error('Model not loaded');
  const titleText = 'Represent this sentence: ' + title.trim().toLowerCase();
  const bodyText = 'Represent this sentence: ' + body.trim().toLowerCase().slice(0, 2000);
  const hash = await sha256(titleText + '|||' + bodyText);
  const existing = await idb.get('embeddings', id);
  if (existing && existing.hash === hash) {
    await idb.put('embeddings', { ...existing, last_seen_active: lastSeenActive || Date.now() });
    return { status: 'skipped', id };
  }
  const titleEmb = await embedder(titleText, { pooling: 'mean', normalize: true });
  const bodyEmb = await embedder(bodyText, { pooling: 'mean', normalize: true });
  const tv = titleEmb.data, bv = bodyEmb.data;
  const combined = new Float32Array(tv.length);
  for (let i = 0; i < tv.length; i++) {
    combined[i] = TITLE_WEIGHT * tv[i] + BODY_WEIGHT * bv[i];
  }
  let norm = 0;
  for (let i = 0; i < combined.length; i++) norm += combined[i] * combined[i];
  norm = Math.sqrt(norm);
  if (norm > 0) for (let i = 0; i < combined.length; i++) combined[i] /= norm;
  await idb.put('embeddings', {
    id, vector: Array.from(combined), hash,
    created_at: existing ? existing.created_at : Date.now(),
    last_seen_active: lastSeenActive || Date.now(),
    item_type: itemType
  });
  return { status: 'embedded', id };
}

/* ── NIGHTLY PASS ───────────────────────────────────────────── */

async function runSimilarityPass() {
  const now = Date.now();
  const activeCutoff = now - (DECAY_ACTIVE_DAYS * DAY_MS);
  const linkExpiryCutoff = now - (LINK_EXPIRY_DAYS * DAY_MS);
  const silentCutoff = now - (SILENT_PERIOD_HOURS * 3600000);
  const allEmbeddings = await idb.getAll('embeddings');
  const active = allEmbeddings.filter(e => e.last_seen_active > activeCutoff);
  if (active.length < 2) return { linksFound: 0, compared: 0 };
  const recentCutoff = now - (RECENT_DAYS * DAY_MS);
  const setA = active.filter(e => e.last_seen_active > recentCutoff && e.created_at < silentCutoff);
  const setB = active;
  let linksFound = 0, compared = 0;
  for (const a of setA) {
    const vecA = new Float32Array(a.vector);
    for (const b of setB) {
      if (a.id === b.id) continue;
      const score = cosine(vecA, new Float32Array(b.vector));
      const finalScore = score * (a.item_type === 'note' || b.item_type === 'note' ? SPARK_BOOST : 1.0);
      compared++;
      if (finalScore >= SIMILARITY_THRESHOLD) {
        const linkId = [a.id, b.id].sort().join('_');
        await idb.put('links', { id: linkId, a: a.id, b: b.id, score: Math.round(finalScore * 1000) / 1000, created_at: now });
        linksFound++;
      }
    }
  }
  await idb.deleteOlderThan('links', 'created_at', linkExpiryCutoff);
  for (const e of active) {
    await idb.put('embeddings', { ...e, last_seen_active: now });
  }
  lastPassTime = now;
  return { linksFound, compared, activeCount: active.length };
}

/* ── GET LINKS ──────────────────────────────────────────────── */

async function getLinksForDiscourse(discourseId) {
  const all = await idb.getAll('links');
  return all.filter(l => l.a === discourseId || l.b === discourseId)
    .map(l => ({ linkedId: l.a === discourseId ? l.b : l.a, score: l.score }));
}
async function getAllActiveLinks() { return await idb.getAll('links'); }

/* ── PRUNE ──────────────────────────────────────────────────── */

async function pruneDiscourse(id) {
  await idb.delete('embeddings', id);
  const all = await idb.getAll('links');
  for (const link of all) { if (link.a === id || link.b === id) await idb.delete('links', link.id); }
  return { pruned: id };
}

/* ── MESSAGE HANDLER ────────────────────────────────────────── */

self.onmessage = async (e) => {
  const { id, type, data } = e.data;
  const reply = (result, error) => self.postMessage({ id, type: type + '_REPLY', result, error: error?.message });
  try {
    switch (type) {
      case 'EMBED': {
        if (!isReady) { reply(null, new Error('NOT_READY')); return; }
        reply(await embedDiscourse(data.id, data.title, data.body, data.itemType, data.lastSeenActive));
        break;
      }
      case 'RUN_PASS': {
        if (!isReady) { reply(null, new Error('NOT_READY')); return; }
        reply(await runSimilarityPass());
        break;
      }
      case 'GET_LINKS': {
        if (!watcherDB) { reply([]); return; }
        if (data && data.discourseId) { reply(await getLinksForDiscourse(data.discourseId)); }
        else { reply(await getAllActiveLinks()); }
        break;
      }
      case 'PRUNE': {
        if (!watcherDB) { reply(null); return; }
        reply(await pruneDiscourse(data.id));
        break;
      }
      case 'GET_STATS': {
        if (!watcherDB) { reply({ embeddings: 0, links: 0, ready: false }); return; }
        const embs = await idb.getAll('embeddings');
        const lnks = await idb.getAll('links');
        reply({ embeddings: embs.length, links: lnks.length, lastPass: lastPassTime, ready: isReady });
        break;
      }
      default: reply(null, new Error('UNKNOWN: ' + type));
    }
  } catch (err) { self.postMessage({ id, type: 'ERROR', result: null, error: err.message }); }
};

/* ── BOOT ───────────────────────────────────────────────────── */

async function init() {
  try {
    watcherDB = await openWatcherDB();
    let loaded = await loadModel();
    if (!loaded) { self.postMessage({ type: 'ERROR', reason: 'MODEL_LOAD_FAIL' }); return; }
    isReady = true;
    self.postMessage({ type: 'READY' });
  } catch (err) { self.postMessage({ type: 'ERROR', reason: err.message }); }
}
init();