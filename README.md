# EmotionSense — Speech Emotion Recognition

An end-to-end speech emotion recognition system: dataset → features → classical
baseline → deep learning model → FastAPI backend → React Native mobile app →
landing page.

Full write-up (problem statement, dataset, methodology, results, confusion
matrices) lives in **[docs/README.md](docs/README.md)**. This file is the
map + the fastest path to running everything yourself.

## Project layout

```
data/               raw + processed RAVDESS data (gitignored — see Setup)
ml/                 dataset prep, feature extraction, training, evaluation, inference
  models/           trained model artifacts (committed, ~12MB — backend runs without retraining)
backend/            FastAPI inference server (/predict, /predict-live, /health)
mobile/             Expo / React Native app
website/            landing page (self-contained index.html)
docs/               write-up + results/ (confusion matrices, metrics, comparison table)
```

## Quick start

### 1. ML pipeline (dataset → trained models)

```bash
cd ml
pip install -r requirements.txt

python data_prep.py                # download RAVDESS separately, see docs/README.md#2-dataset--preprocessing
python feature_extraction.py       # caches classical features + spectrograms
python augment_train.py            # augmented spectrograms for the CNN (train split only)
python train_baseline.py           # Random Forest + SVM, GridSearchCV
python train_deep.py               # CNN on mel-spectrograms
python evaluate.py                 # confusion matrices, per-class metrics, comparison table
python quantize_model.py           # optional: TFLite export for constrained deployment
```

Every step caches its output, so re-running a script after the first successful run is a no-op
unless you pass `--force`.

### 2. Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

`POST /predict` with a multipart `file` field returns:

```json
{
  "emotion": "happy",
  "confidence": 0.81,
  "probabilities": { "angry": 0.03, "happy": 0.81, "...": "..." },
  "model_used": "cnn",
  "duration_ms": 142.3
}
```

### 3. Mobile app

```bash
cd mobile
npm install
npx expo start
```

See [mobile/README.md](mobile/README.md) for pointing the app at a deployed backend.

### 4. Website

`website/index.html` is a single self-contained file (fonts inlined, no build step, no
external requests) — open it directly or deploy it to any static host.

## Deployment

See [docs/README.md#7-deployment](docs/README.md#7-deployment) for Render/Docker instructions and
the model-quantization trade-offs for free-tier hosting.
