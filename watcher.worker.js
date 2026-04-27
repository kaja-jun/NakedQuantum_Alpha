/**

- WATCHER v1.0 -- NakedQuantum
- 
- The silent observer. Finds semantic connections between your discourses.
- Never speaks unless it sees something. Never runs on keystroke.
- 
- Model: Xenova/bge-small-en-v1.5 (133MB, cached once in Cache API)
- Storage: IndexedDB ‘nq_watcher’ -- separate from encrypted SQLite
- Similarity: Vanilla JS cosine on Float32Array -- no sqlite-vec needed
- 
- BGE-Small chosen over MiniLM because:
- - Trained for asymmetric reasoning, not just synonym matching
- - Handles philosophical nuance, contradiction detection
- - 130MB downloads once, lives in Cache API forever
- - For 300 notes of a life, the marginal gains ARE the point
    */

/* ============================================================

1. BOOTSTRAP -- Load Transformers.js + Open IndexedDB
   ============================================================ */

const MODEL_ID = ‘Xenova/bge-small-en-v1.5’;
const MODEL_CACHE_KEY = ‘watcher-model-v1’;
const DB_NAME = ‘nq_watcher’;
const DB_VERSION = 1;
const SIMILARITY_THRESHOLD = 0.82;
const DECAY_ACTIVE_DAYS = 25;
const LINK_EXPIRY_DAYS = 14;
const DAY_MS = 86400000;

let embedder = null;
let watcherDB = null;
let isReady = false;
let lastPassTime = 0;

/* ============================================================
2. INDEXEDDB -- Separate from encrypted SQLite, rebuildable
============================================================ */

function openWatcherDB() {
return new Promise((resolve, reject) => {
const req = indexedDB.open(DB_NAME, DB_VERSION);

```
req.onupgradeneeded = (e) => {
  const db = e.target.result;

  // embeddings store -- one per discourse
  if (!db.objectStoreNames.contains('embeddings')) {
    const emb = db.createObjectStore('embeddings', { keyPath: 'id' });
    emb.createIndex('last_seen_active', 'last_seen_active', { unique: false });
  }

  // links store -- similarity pairs found by nightly pass
  if (!db.objectStoreNames.contains('links')) {
    const links = db.createObjectStore('links', { keyPath: 'id' });
    links.createIndex('a', 'a', { unique: false });
    links.createIndex('b', 'b', { unique: false });
    links.createIndex('created_at', 'created_at', { unique: false });
  }
};

req.onsuccess = (e) => resolve(e.target.result);
req.onerror = (e) => reject(e.target.error);
```

});
}

const idb = {
get: (store, key) => new Promise((res) => {
const req = watcherDB.transaction(store, ‘readonly’).objectStore(store).get(key);
req.onsuccess = () => res(req.result || null);
req.onerror = () => res(null);
}),

put: (store, obj) => new Promise((res, rej) => {
const req = watcherDB.transaction(store, ‘readwrite’).objectStore(store).put(obj);
req.onsuccess = () => res();
req.onerror = (e) => rej(e.target.error);
}),

delete: (store, key) => new Promise((res) => {
const req = watcherDB.transaction(store, ‘readwrite’).objectStore(store).delete(key);
req.onsuccess = () => res();
req.onerror = () => res();
}),

getAll: (store) => new Promise((res) => {
const req = watcherDB.transaction(store, ‘readonly’).objectStore(store).getAll();
req.onsuccess = () => res(req.result || []);
req.onerror = () => res([]);
}),

getAllByIndex: (store, index, value) => new Promise((res) => {
const req = watcherDB.transaction(store, ‘readonly’)
.objectStore(store)
.index(index)
.getAll(IDBKeyRange.only(value));
req.onsuccess = () => res(req.result || []);
req.onerror = () => res([]);
}),

deleteOlderThan: (store, indexName, cutoff) => new Promise((res) => {
const range = IDBKeyRange.upperBound(cutoff);
const req = watcherDB.transaction(store, ‘readwrite’)
.objectStore(store)
.index(indexName)
.openCursor(range);
req.onsuccess = (e) => {
const cursor = e.target.result;
if (cursor) { cursor.delete(); cursor.continue(); }
else res();
};
req.onerror = () => res();
})
};

/* ============================================================
3. MODEL -- BGE-Small, cached in Cache API, loaded once
============================================================ */

async function loadModel() {
try {
// Transformers.js v3 via CDN -- loaded via importScripts below
const { pipeline, env } = await import(
‘https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/dist/transformers.min.js’
);

```
// Use Cache API for model weights -- survives worker restarts
env.useBrowserCache = true;
env.cacheDir = MODEL_CACHE_KEY;

// WebGPU if available (iPhone 14 has it), fallback to WASM
env.backends.onnx.wasm.proxy = false;

embedder = await pipeline('feature-extraction', MODEL_ID, {
  quantized: true,
  progress_callback: (progress) => {
    if (progress.status === 'downloading') {
      self.postMessage({
        type: 'MODEL_PROGRESS',
        loaded: Math.round(progress.loaded / 1024 / 1024),
        total: Math.round(progress.total / 1024 / 1024)
      });
    }
  }
});

return true;
```

} catch (err) {
console.error(’[Watcher] Model load failed:’, err);
return false;
}
}

/* ============================================================
4. SHA256 HASH -- Skip recompute if text unchanged
============================================================ */

async function sha256(text) {
const buf = await crypto.subtle.digest(
‘SHA-256’,
new TextEncoder().encode(text)
);
return Array.from(new Uint8Array(buf))
.map(b => b.toString(16).padStart(2, ‘0’))
.join(’’);
}

/* ============================================================
5. COSINE SIMILARITY -- Pure Float32Array math, zero allocations
============================================================ */

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

/* ============================================================
6. EMBED -- One discourse at a time, on save only
============================================================ */

async function embedDiscourse(id, text, lastSeenActive) {
if (!embedder) throw new Error(‘Model not loaded’);

// Normalize: BGE-Small needs "Represent this sentence:" prefix for asymmetric search
const normalized = ’Represent this sentence: ’ + text.trim().toLowerCase().slice(0, 2000);
const hash = await sha256(normalized);

// Check if we already have this exact content
const existing = await idb.get(‘embeddings’, id);
if (existing && existing.hash === hash) {
// Text unchanged -- just bump last_seen_active
await idb.put(‘embeddings’, {
…existing,
last_seen_active: lastSeenActive || Date.now()
});
return { status: ‘skipped’, id };
}

// Run embedding
const output = await embedder(normalized, { pooling: ‘mean’, normalize: true });
const vector = Array.from(output.data); // Store as regular array for IndexedDB serialization

await idb.put(‘embeddings’, {
id,
vector,
hash,
created_at: existing ? existing.created_at : Date.now(),
last_seen_active: lastSeenActive || Date.now()
});

return { status: ‘embedded’, id };
}

/* ============================================================
7. NIGHTLY PASS -- Similarity comparison, once per day
============================================================ */

async function runSimilarityPass() {
const now = Date.now();
const activeCutoff = now - (DECAY_ACTIVE_DAYS * DAY_MS);
const linkExpiryCutoff = now - (LINK_EXPIRY_DAYS * DAY_MS);

// Load all active embeddings (not decayed past 25 days)
const allEmbeddings = await idb.getAll(‘embeddings’);
const active = allEmbeddings.filter(e => e.last_seen_active > activeCutoff);

if (active.length < 2) {
return { linksFound: 0, compared: 0 };
}

// Set A = updated in last 7 days (the "fresh thoughts")
const recentCutoff = now - (7 * DAY_MS);
const setA = active.filter(e => e.last_seen_active > recentCutoff);
// Set B = all active (compare fresh against everything)
const setB = active;

let linksFound = 0;
let compared = 0;

for (const a of setA) {
const vecA = new Float32Array(a.vector);

```
for (const b of setB) {
  if (a.id === b.id) continue;

  const vecB = new Float32Array(b.vector);
  const score = cosine(vecA, vecB);
  compared++;

  if (score >= SIMILARITY_THRESHOLD) {
    // Canonical link ID -- sorted so a_b and b_a are the same link
    const linkId = [a.id, b.id].sort().join('_');

    await idb.put('links', {
      id: linkId,
      a: a.id,
      b: b.id,
      score: Math.round(score * 1000) / 1000, // 3 decimal places
      created_at: now
    });

    linksFound++;
  }
}
```

}

// Prune expired links
await idb.deleteOlderThan(‘links’, ‘created_at’, linkExpiryCutoff);

// Update last_seen_active for all active embeddings
for (const e of active) {
await idb.put(‘embeddings’, { …e, last_seen_active: now });
}

lastPassTime = now;

return { linksFound, compared, activeCount: active.length };
}

/* ============================================================
8. GET LINKS FOR DISCOURSE -- Called by main thread for UI dot
============================================================ */

async function getLinksForDiscourse(discourseId) {
const allLinks = await idb.getAll(‘links’);
const relevant = allLinks.filter(l => l.a === discourseId || l.b === discourseId);

return relevant.map(l => ({
linkedId: l.a === discourseId ? l.b : l.a,
score: l.score
}));
}

async function getAllActiveLinks() {
return await idb.getAll(‘links’);
}

/* ============================================================
9. PRUNE -- Called when discourse deleted from Deep Soup
============================================================ */

async function pruneDiscourse(id) {
await idb.delete(‘embeddings’, id);

// Remove all links involving this discourse
const allLinks = await idb.getAll(‘links’);
for (const link of allLinks) {
if (link.a === id || link.b === id) {
await idb.delete(‘links’, link.id);
}
}

return { pruned: id };
}

/* ============================================================
10. INIT -- Boot sequence
============================================================ */

async function init() {
try {
watcherDB = await openWatcherDB();

```
const modelLoaded = await loadModel();

if (!modelLoaded) {
  // Retry once after 5 seconds
  await new Promise(r => setTimeout(r, 5000));
  const retryLoad = await loadModel();
  if (!retryLoad) {
    self.postMessage({ type: 'ERROR', reason: 'MODEL_LOAD_FAIL' });
    return;
  }
}

isReady = true;
self.postMessage({ type: 'READY' });
```

} catch (err) {
console.error(’[Watcher] Boot failed:’, err);
self.postMessage({ type: ‘ERROR’, reason: err.message });
}
}

/* ============================================================
11. MESSAGE HANDLER -- Fire-and-forget protocol
============================================================ */

self.onmessage = async (e) => {
const { id, type, data } = e.data;

const reply = (result, error) => {
self.postMessage({ id, type: type + ‘_REPLY’, result, error: error?.message });
};

try {
switch (type) {

```
  case 'EMBED': {
    // Called after every discourse save
    // data = { id, text, lastSeenActive }
    if (!isReady) { reply(null, new Error('NOT_READY')); return; }
    const result = await embedDiscourse(data.id, data.text, data.lastSeenActive);
    reply(result);
    break;
  }

  case 'RUN_PASS': {
    // Called once per day from main thread (foreground, hour > 3, lastRun > 24h)
    if (!isReady) { reply(null, new Error('NOT_READY')); return; }
    const passResult = await runSimilarityPass();
    reply(passResult);
    break;
  }

  case 'GET_LINKS': {
    // Called by main thread to check if Watcher dot should appear
    // data = { discourseId } -- or omit for all links
    if (!watcherDB) { reply([]); return; }
    if (data?.discourseId) {
      const links = await getLinksForDiscourse(data.discourseId);
      reply(links);
    } else {
      const allLinks = await getAllActiveLinks();
      reply(allLinks);
    }
    break;
  }

  case 'PRUNE': {
    // Called when discourse deleted from Deep Soup permanently
    // data = { id }
    if (!watcherDB) { reply(null); return; }
    const pruneResult = await pruneDiscourse(data.id);
    reply(pruneResult);
    break;
  }

  case 'UPDATE_ACTIVE': {
    // Called on app foreground -- bump last_seen_active for all visible discourses
    // data = { ids: [...] }
    if (!watcherDB) { reply(null); return; }
    const now = Date.now();
    for (const discId of (data?.ids || [])) {
      const existing = await idb.get('embeddings', discId);
      if (existing) {
        await idb.put('embeddings', { ...existing, last_seen_active: now });
      }
    }
    reply({ updated: data?.ids?.length || 0 });
    break;
  }

  case 'GET_STATS': {
    // Dev/debug: how many embeddings and links do we have
    if (!watcherDB) { reply({ embeddings: 0, links: 0 }); return; }
    const embs = await idb.getAll('embeddings');
    const lnks = await idb.getAll('links');
    reply({ embeddings: embs.length, links: lnks.length, lastPass: lastPassTime, ready: isReady });
    break;
  }

  default:
    reply(null, new Error('UNKNOWN_MESSAGE_TYPE: ' + type));
}
```

} catch (err) {
console.error(’[Watcher] Handler error:’, err);
reply(null, err);
}
};

/* ============================================================
12. BOOT
============================================================ */

init();