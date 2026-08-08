// Local history persistence (AsyncStorage). Each analysis is stored as a
// self-contained record so History / Result screens don't need network
// round-trips to re-render past results.
import AsyncStorage from '@react-native-async-storage/async-storage';

const HISTORY_KEY = '@ser/history';

export async function getHistory() {
  const raw = await AsyncStorage.getItem(HISTORY_KEY);
  if (!raw) return [];
  try {
    return JSON.parse(raw);
  } catch (_) {
    return [];
  }
}

export async function saveAnalysis(record) {
  const history = await getHistory();
  const entry = {
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    timestamp: new Date().toISOString(),
    ...record,
  };
  const updated = [entry, ...history].slice(0, 200); // cap local history
  await AsyncStorage.setItem(HISTORY_KEY, JSON.stringify(updated));
  return entry;
}

export async function deleteAnalysis(id) {
  const history = await getHistory();
  const updated = history.filter((h) => h.id !== id);
  await AsyncStorage.setItem(HISTORY_KEY, JSON.stringify(updated));
  return updated;
}

export async function clearHistory() {
  await AsyncStorage.removeItem(HISTORY_KEY);
}
