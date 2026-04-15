self.addEventListener("install", () => {
  console.log("Installed");
});

self.addEventListener("fetch", (event) => {
  event.respondWith(fetch(event.request));
});
