# Clarifications: Demucs Stem Server

### Q: How is the API key transmitted on the wire?

**A:** [OPEN] — `--api-key` and `FEEDBACK_API_KEY` are accepted but
the README and code comments do not document the request header. Most
likely a custom `X-API-Key` header or `Authorization: Bearer <key>`;
needs to be pinned and documented.

### Q: Does the cache invalidate when `--model` changes?

**A:** [OPEN] — the cache key is based on audio hash + stem set; the
model is not part of the key in `server.py`. Switching from
`htdemucs_ft` to `htdemucs_6s` may either return the previously cached
4-stem output or coexist depending on stem set differences. Behaviour
should be specified — and the model probably belongs in the cache key.

### Q: Is there an in-flight de-duplication for two clients hitting
`/separate` with the same audio at the same time?

**A:** [OPEN] — the cache short-circuit happens at job-create time. If
two requests arrive simultaneously and neither has a cached result
yet, both probably enqueue and run Demucs twice. A future fix would
hash-lock around job creation.

### Q: What is the maximum upload size?

**A:** [OPEN] — there is no explicit cap in `server.py`. The feedBack
side likely throws something reasonable in front of it, but as a
standalone service the server should probably enforce its own ceiling.

### Q: Is there a default Demucs model override per user, or per
request?

**A:** Per request via the `model` query parameter on `/separate`;
default is whatever was passed to `--model` at boot, which itself
defaults to `htdemucs_ft`. No per-user state.

### Q: When the LRU evicts the English wav2vec2 aligner, what happens
to in-flight `/align en` calls?

**A:** They complete using the loaded aligner — eviction only applies
to the cache pointer, not to refs already held inside a running
request. Subsequent `/align en` calls trigger a reload, and
`/health.warmup.whisperx == "evicted"` reports it.

### Q: Where do the pre-baked Lyrics-Karaoke / Lyrics-Sync inputs in
the demo come from?

**A:** The demo (`feedBack-demo`) bundles a `.sloppak` that already
contains `lyrics.json` + `vocal_pitch.json`. The demo Dockerfile does
not run a Demucs server — see `feedBack-demo/README.md` "What's
blocked".

### Q: What test coverage exists?

**A:** None in the repo today. Constitution Principle V (CUDA hygiene)
and the pitch / align post-processors are the obvious candidates for
a future pytest suite.

### Q: Can the server be reached over plain HTTP from outside the
local network?

**A:** Yes by default (`--host 0.0.0.0`, CORS `allow_origins=*`). For
public exposure, run behind a reverse proxy with TLS and set
`--api-key`. The current configuration assumes a trusted home network.

### Q: How do new languages get pre-warmed?

**A:** They don't — by design. Only English wav2vec2 is part of the
warmup contract because we don't know which language a client will
ask for. Other languages download on first `/align` call. Clients can
poll `/health.warmup.whisperx_aligners[lang]` to wait for readiness
before issuing real traffic.
