# Deployment Guide

Three separate things get deployed, to three separate places. Do them in this order — the
website's "Start Dialogue" flow and the mobile app both need the backend's URL, so get that
one first.

```
GitHub (source of truth)
   ├── Render   -> backend/  (FastAPI + models, Docker)
   ├── Vercel   -> website/  (static index.html, no build)
   └── (later)  -> mobile/   (Expo, separate app-store pipeline)
```

## 1. Push to GitHub

1. Go to [github.com/new](https://github.com/new) and create an **empty** repository — don't
   check "Add a README", "Add .gitignore", or "Add a license" (this repo already has all three;
   letting GitHub create its own would conflict with the first push).
2. Copy the repo URL it gives you (looks like `https://github.com/<you>/<repo>.git`).
3. From the project root:

   ```bash
   cd "VOICE EMOTION"
   git remote add origin https://github.com/<you>/<repo>.git
   git branch -M main
   git push -u origin main
   ```

   (This repo's first commit is currently on a branch named `master` locally — `git branch -M main`
   renames it to `main` before pushing, matching what Render/Vercel expect by default.)

If push is rejected for being too large: it shouldn't be (the whole repo is ~5MB compressed,
`ml/models/` is the biggest thing in it at ~12MB uncompressed) — but if you've since retrained with
a bigger model, see the note in `docs/README.md#7-deployment` about pulling large artifacts from a
release/bucket instead of committing them.

## 2. Backend → Render

1. Sign in at [render.com](https://render.com) (GitHub OAuth is the fastest option — it also grants
   Render permission to read your repos, which the next step needs).
2. **New +** → **Blueprint**.
3. Pick the repo you just pushed. Render auto-detects `render.yaml` at the repo root — no path
   configuration needed, it's placed there on purpose.
4. Review the one service it finds (`emotionsense-api`, Docker, free plan) and click **Apply**.
5. First build takes a few minutes (installing TensorFlow is the slow part). Once it's live, open
   the service's `.onrender.com` URL + `/health` — you should get back
   `{"status":"ok","model_backend":"cnn",...}`.

**Free-tier note**: Render's free web services spin down after 15 minutes of no traffic and take
~30-60s to wake back up on the next request. That's normal, not a bug — if you need always-on,
that's a paid plan on Render or Railway.

**If you retrain and want to swap in a smaller model**: `ml/quantize_model.py` produces
`ml/models/cnn_model.tflite` (0.12MB vs. the Keras model's 1.4MB) for exactly this situation —
see `docs/results/quantization_results.json` for the accuracy trade-off before switching.

## 3. Website → Vercel

The site is one self-contained `website/index.html` — no `package.json`, no build step, no
external requests (fonts are inlined as base64). This makes it about as simple a Vercel deploy as
exists.

1. Sign in at [vercel.com](https://vercel.com) (GitHub OAuth again).
2. **Add New** → **Project** → import the same GitHub repo.
3. Set **Root Directory** to `website`.
4. Framework preset: **Other** (there's no framework — it's a static HTML file). Leave the build
   command empty; output directory `.`.
5. Deploy. Vercel gives you a `*.vercel.app` URL immediately.

(Netlify, GitHub Pages, or Cloudflare Pages all work identically for this file — Vercel isn't
required, just convenient since you're likely already using it for the GitHub OAuth flow above.)

## 4. Wire the deployed backend into the frontend

Once Render gives you a real backend URL (e.g. `https://emotionsense-api.onrender.com`):

**Website** — open `website/index.html` and update the two placeholder links in the
`#start-dialogue` section (currently `href="#"`):

```html
<a href="https://emotionsense-api.onrender.com/docs" class="btn-pill">Read the Docs</a>
<a href="https://github.com/<you>/<repo>" class="btn-ghost">View on GitHub</a>
```

(`/docs` is FastAPI's built-in interactive Swagger UI — free API documentation with zero extra work.)
Redeploy by pushing the change; Vercel auto-redeploys on every push to `main`.

**Mobile app** — edit `mobile/app.json`:

```json
"extra": { "apiUrl": "https://emotionsense-api.onrender.com" }
```

or set `EXPO_PUBLIC_API_URL` when running `npx expo start`. See `mobile/README.md` for the
localhost-vs-LAN-IP nuance when testing on a physical device against a local backend.

## 5. Mobile app distribution (optional, separate pipeline)

Everything above gets you a live backend + website. The mobile app is a bigger, separate step
because it needs Apple/Google developer accounts and store review, not just a `git push`:

- **Quick testing, no store**: `cd mobile && npx expo start`, scan the QR with Expo Go. Works
  immediately, no account needed, but requires Expo Go installed on the test device.
- **Real app-store build**: [EAS Build](https://docs.expo.dev/build/introduction/)
  (`npx eas build`) — needs a free Expo account (no payment), plus a paid Apple Developer account
  ($99/yr) for iOS and a one-time $25 Google Play Console fee for Android. This is a genuinely
  separate decision from the web/backend deploy above — worth doing once the backend URL is
  final and stable, not before.
