// Talks to the FastAPI backend (see /backend/main.py).
// Point EXPO_PUBLIC_API_URL at your deployed backend, e.g. via app.json "extra"
// or a .env consumed by babel-plugin-dotenv-import; defaults to localhost for
// the Expo Go / simulator dev loop.
import Constants from 'expo-constants';

const DEFAULT_API_URL = 'http://localhost:8000';

export function getApiUrl() {
  return (
    process.env.EXPO_PUBLIC_API_URL ||
    Constants.expoConfig?.extra?.apiUrl ||
    DEFAULT_API_URL
  );
}

/**
 * Uploads a recorded/picked audio file for emotion inference.
 * @param {{ uri: string, name?: string, mimeType?: string }} file
 * @returns {Promise<{ emotion: string, confidence: number, probabilities: object, model_used: string, duration_ms: number }>}
 */
export async function predictEmotion(file) {
  const formData = new FormData();
  formData.append('file', {
    uri: file.uri,
    name: file.name || 'clip.wav',
    type: file.mimeType || 'audio/wav',
  });

  const res = await fetch(`${getApiUrl()}/predict`, {
    method: 'POST',
    body: formData,
    headers: { 'Content-Type': 'multipart/form-data' },
  });

  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch (_) {
      // ignore, keep generic message
    }
    throw new Error(detail);
  }
  return res.json();
}

export async function checkHealth() {
  try {
    const res = await fetch(`${getApiUrl()}/health`, { method: 'GET' });
    return res.ok ? res.json() : { status: 'unreachable' };
  } catch (e) {
    return { status: 'unreachable', detail: e.message };
  }
}
