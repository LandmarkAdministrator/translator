/* Service worker for the live translation page.
 *
 * Deliberately minimal. This app's whole value is being LIVE, so nothing that
 * carries service content is ever served from cache — only the shell (page,
 * manifest, icons) is cached so the app opens instantly and shows a sensible
 * screen when someone launches it outside a service or on a bad connection.
 *
 * WebSocket traffic is not touched by service workers at all, so the live
 * stream is unaffected either way.
 */
const SHELL = "translation-shell-v1";
const SHELL_FILES = [
  "/",
  "/manifest.webmanifest",
  "/icons/icon-192.png",
  "/icons/icon-512.png",
];

self.addEventListener("install", (event) => {
  // Take over promptly: a stale shell after an update is worse than a reload.
  self.skipWaiting();
  event.waitUntil(caches.open(SHELL).then((c) => c.addAll(SHELL_FILES)).catch(() => {}));
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== SHELL).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (event) => {
  const req = event.request;
  if (req.method !== "GET") return;
  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return;

  // Network first, falling back to the cached shell when offline. Always
  // prefer the live network copy so an updated page is picked up promptly.
  event.respondWith(
    fetch(req)
      .then((res) => {
        if (res && res.ok && SHELL_FILES.includes(url.pathname)) {
          const copy = res.clone();
          caches.open(SHELL).then((c) => c.put(req, copy)).catch(() => {});
        }
        return res;
      })
      .catch(() => caches.match(req).then((hit) => hit || caches.match("/")))
  );
});
