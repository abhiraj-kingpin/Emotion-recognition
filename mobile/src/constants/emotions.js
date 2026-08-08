// Mirrors ml/utils.py EMOTIONS / EMOTION_COLORS so the whole stack (model
// output, backend, mobile app, website) agrees on label order and color.
export const EMOTIONS = [
  'angry',
  'calm',
  'disgust',
  'fearful',
  'happy',
  'neutral',
  'sad',
  'surprised',
];

export const EMOTION_META = {
  angry: { color: '#E5484D', icon: 'flame', label: 'Angry' },
  calm: { color: '#4CC9B0', icon: 'leaf', label: 'Calm' },
  disgust: { color: '#8B7A3F', icon: 'sad-outline', label: 'Disgust' },
  fearful: { color: '#7C6FE0', icon: 'skull-outline', label: 'Fearful' },
  happy: { color: '#F5C542', icon: 'sunny', label: 'Happy' },
  neutral: { color: '#9AA5B1', icon: 'ellipse-outline', label: 'Neutral' },
  sad: { color: '#4A7FC9', icon: 'rainy', label: 'Sad' },
  surprised: { color: '#E8823B', icon: 'flash', label: 'Surprised' },
};

export function emotionColor(emotion) {
  return EMOTION_META[emotion]?.color ?? '#9AA5B1';
}

export function emotionLabel(emotion) {
  return EMOTION_META[emotion]?.label ?? emotion;
}
