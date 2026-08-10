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

`app.json`'s `extra.apiUrl` already points at the deployed Render backend (see `src/services/api.js`
for how it's read). To point at a different backend instead, either:

- edit `extra.apiUrl` in `app.json`, or
- run with `EXPO_PUBLIC_API_URL=https://your-api.onrender.com npx expo start`

If you ever point this at a backend running on your own laptop instead of a deployed one:
`localhost` won't resolve from a physical device — use your laptop's LAN IP
(`http://192.168.x.x:8000`) instead.

## Building a standalone APK (no dev server needed)

Expo Go is a dev-mode viewer only — it has to match this project's exact SDK version, and it needs
a running dev server. For something installable that works standalone, build a real APK via
[EAS Build](https://docs.expo.dev/build/introduction/) (Expo's cloud build service — no Android
Studio needed):

```bash
cd mobile
npx eas login          # needs a free Expo account (expo.dev) - browser login
npx eas build:configure  # links this project to your Expo account, first time only
npx eas build --platform android --profile preview
```

That queues a cloud build and gives you a download link when it's done (a few minutes) — the
`preview` profile in `eas.json` is set to build a directly-installable `.apk` (not the Play-Store-only
`.aab` format `production` uses). Download the link on your phone and install it directly (you'll
need to allow "install from unknown sources" once).

## Notes

- Recording uses `expo-av` with metering enabled; the orb and waveform are driven directly off
  the live dBFS reading, not a canned animation.
- History is stored locally via `@react-native-async-storage/async-storage` — nothing is
  uploaded except the audio clip sent to `/predict` for inference.
- `App.js` wraps everything in `GestureHandlerRootView` — required for touches to register
  reliably with `react-native-screens`' native-stack navigator. Easy to forget, and the failure
  mode if you do (buttons silently not responding) doesn't point at the cause at all.
