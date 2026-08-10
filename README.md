# EmotionSense

A speech emotion recognition system I built end to end — from raw audio to a
trained model to something you can actually talk to. Give it a few seconds of
speech and it tells you what emotion it's hearing: angry, calm, disgust,
fearful, happy, neutral, sad, or surprised.

It's not just a notebook with a model in it. There's a real training
pipeline, a baseline I could compare against, a backend serving live
predictions, a mobile app, and a website you can try it on right now.

**Live demo:** [emotion-recognition-sable.vercel.app](https://emotion-recognition-sable.vercel.app) — click "Start Dialogue" and either upload a clip or just talk into your mic.

## What's actually here

```
data/               RAVDESS dataset, raw + processed (gitignored, see below)
ml/                 everything from parsing filenames to a trained model
  models/           the trained weights themselves, checked in so the backend
                     just works without anyone having to retrain first
backend/            FastAPI server that actually runs the model
mobile/             Expo app — record on your phone, get an emotion back
website/            the landing page + in-browser demo
docs/               the full write-up, if you want the details
```

## The short version

I trained on [RAVDESS](https://zenodo.org/records/1188976) — 1,440 clips of
actors reading the same two lines in eight different emotional deliveries.
Started with a Random Forest and an SVM on hand-picked audio features
(MFCCs, chroma, that kind of thing) as a baseline — got those to about 58%
accuracy with real grid search, not guessed hyperparameters. Then built a
CNN that reads the mel-spectrogram like an image instead, which pushed
accuracy up to 66.7%. Not a huge jump, but a real one, and I'd rather report
an honest 8-point improvement than pretend either number is higher than it
is.

Both models get confused in the same two places, which I think is more
interesting than either model's individual quirks: calm and sad get mixed
up constantly (both are low-energy, low-pitch-variance speech — genuinely
hard to tell apart from tone alone), and angry/happy cross over sometimes
too since both are high-energy deliveries. I'd rather write that down than
hide it behind a cherry-picked confusion matrix.

Full breakdown — dataset, features, architecture, results, what I'd try
next — is in [docs/README.md](docs/README.md).

## Running it yourself

**The model pipeline:**

```bash
cd ml
pip install -r requirements.txt

python data_prep.py                # parses RAVDESS filenames into labels + splits
python feature_extraction.py       # MFCCs, chroma, mel-spectrograms — cached to disk
python augment_train.py            # pitch/time/noise augmentation, training split only
python train_baseline.py           # Random Forest + SVM, GridSearchCV
python train_deep.py               # the CNN
python evaluate.py                 # confusion matrices, per-class scores, comparison table
python quantize_model.py           # optional — shrinks the model for tighter hosting
```

Everything caches what it computes, so running a step twice just reuses the
last result unless you pass `--force`.

**The backend:**

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

Send it audio, get back an emotion:

```json
{
  "emotion": "happy",
  "confidence": 0.81,
  "probabilities": { "angry": 0.03, "happy": 0.81, "...": "..." },
  "model_used": "cnn",
  "duration_ms": 142.3
}
```

**The mobile app:**

```bash
cd mobile
npm install
npx expo start
```

More detail — including how to point it at your own backend, and how to
build an actual installable APK instead of running through Expo Go — is in
[mobile/README.md](mobile/README.md).

**The website** is a single HTML file — no build step, no dependencies, fonts
baked in as base64 so it doesn't even need the internet for those. Open it
straight in a browser or drop it on any static host.

## Where it lives

- Backend: Render (free tier — the first request after it's been idle takes
  a while to wake up, that's just how free hosting works, not a bug)
- Website: Vercel
- Both auto-deploy on push to `main`

Deployment steps if you're setting this up fresh are in
[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

## What I'd improve next

- The dataset is all acted, studio-quality speech. Real-world audio (background
  noise, overlapping speakers, bad mics) would knock accuracy down, and I
  haven't tested against that yet.
- Nothing here detects "is this even speech" before classifying it — feed it
  laughter or a pure tone and it'll still confidently pick one of the eight
  emotions, because that's all it knows how to do. A real product needs a
  speech/non-speech gate in front of it.
- Longer training runs and a bigger, more varied dataset are the obvious next
  levers for accuracy, more than further architecture tweaking at this point.
