// KrishiSahay Service Worker — Offline Support
const CACHE_NAME = 'krishisahay-v1';
const STATIC_ASSETS = ['/', '/index.html'];

const OFFLINE_KB = [
  { q: 'PM-KISAN', a: 'PM-KISAN gives ₹6,000/year in 3 installments to all small/marginal farmers. Apply at pmkisan.gov.in or CSC center with Aadhaar + bank passbook + land records.' },
  { q: 'crop insurance', a: 'PMFBY crop insurance: Kharif 2%, Rabi 1.5% premium. Apply within 2 weeks of sowing at your bank or pmfby.gov.in. Claim within 72 hours of damage.' },
  { q: 'kisan credit card', a: 'KCC: Credit at 7% p.a. (4% effective). Apply at any bank. PM-KISAN farmers get it without income proof. Covers seeds, fertilizers, equipment.' },
];

self.addEventListener('install', e => {
  e.waitUntil(caches.open(CACHE_NAME).then(c => c.addAll(STATIC_ASSETS)));
  self.skipWaiting();
});

self.addEventListener('activate', e => {
  e.waitUntil(caches.keys().then(keys =>
    Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k)))
  ));
  self.clients.claim();
});

self.addEventListener('fetch', e => {
  if (e.request.url.includes('/query')) {
    e.respondWith(
      fetch(e.request.clone()).catch(() => {
        return new Response(JSON.stringify({
          answer: 'You are offline. Please check your connection. For urgent help call 1800-180-1551.',
          method: 'offline',
          sources: [],
          related: []
        }), { headers: { 'Content-Type': 'application/json' } });
      })
    );
    return;
  }
  e.respondWith(
    caches.match(e.request).then(cached => cached || fetch(e.request))
  );
});
