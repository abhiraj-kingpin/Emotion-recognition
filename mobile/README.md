# EmotionSense — Mobile App

Expo (React Native) client for the Speech Emotion Recognition backend.

## Screens

| Screen | Purpose |
|---|---|
| **Home** | "New Recording" CTA, Live Record / Upload Audio cards, History / Insights cards, recent-analyses chip row |
| **Recording** | Orb reacts to state — idle (pulse) → recording (amplitude-driven) → analyzing (shimmer) → result (recolors to the detected emotion). Handles both the live-mic flow and the upload flow via `route.params.source`. |
| **Result** | Emotion + confidence, per-class probability breakdown, Re-analyze / View History actions |
| **History** | FlatList of past analyses with emotion filter chips |

Navigation: `Home → (Recording[live] | Recording[upload]) → Result`, `Home → History → Result`.

## Setup

```bash
cd mobile
npm install
npx expo start
```

Scan the QR code with **Expo Go** (Android) or the Camera app (iOS), or press `a` / `i` for an emulator/simulator.

## Pointing at the backend

By default the app calls `http://localhost:8000` (see `src/services/api.js`). To point at a
deployed backend, either:

- set `extra.apiUrl` in `app.json`, or
- run with `EXPO_PUBLIC_API_URL=https://your-api.onrender.com npx expo start`

When testing on a physical device against a backend running on your laptop, `localhost` won't
resolve to your machine — use your LAN IP (`http://192.168.x.x:8000`) instead.

## Notes

- Recording uses `expo-av` with metering enabled; the orb and waveform are driven directly off
  the live dBFS reading, not a canned animation.
- History is stored locally via `@react-native-async-storage/async-storage` — nothing is
  uploaded except the audio clip sent to `/predict` for inference.
- `react-navigation` is pinned to v6 (not v7) because Expo SDK 51's blessed `react-native-screens`
  version doesn't satisfy v7's peer dependency yet.
