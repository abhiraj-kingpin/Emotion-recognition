// Short, practical, non-clinical suggestions shown on the Result screen.
// Deliberately not medical or therapeutic advice — just a small, honest nudge.
// Keep entries short: one line acknowledging what was heard, 2-3 concrete things
// to actually try, nothing preachy.

export const EMOTION_ADVICE = {
  angry: {
    headline: "There's real heat in this one.",
    tips: [
      'Before you respond to anything, take a few slow breaths — even 30 seconds helps.',
      "If you can, step away from whatever triggered it for a bit before deciding what to do.",
      'Naming the actual trigger out loud (or jotting it down) usually takes some of the edge off.',
    ],
  },
  disgust: {
    headline: "Something about this didn't sit right.",
    tips: [
      "Worth asking whether this is about the situation itself, or a boundary that got crossed.",
      'If you can step away from whatever caused it, do — you don\'t have to push through it.',
      "This one often fades faster once you name specifically what bothered you.",
    ],
  },
  fearful: {
    headline: "Sounds like something's got you on edge.",
    tips: [
      'Try grounding: name 5 things you can see, 4 you can hear, 3 you can touch.',
      'Ask yourself if the threat is immediate or something you\'re anticipating — they need different responses.',
      'Saying it out loud to someone else usually makes it smaller than it feels alone.',
    ],
  },
  happy: {
    headline: 'Good energy in this one.',
    tips: [
      "Worth noting what led to this — it's useful to know your own way back to it.",
      'Happiness tends to multiply when you say it out loud to someone.',
    ],
  },
  neutral: {
    headline: 'Pretty even-keeled right now.',
    tips: [
      "Nothing to fix here — this is a good baseline to check back against later.",
    ],
  },
  sad: {
    headline: 'There\'s some real heaviness in this one.',
    tips: [
      'This deserves space, not a quick fix — it\'s okay to just feel it instead of pushing past it.',
      'Water, food, and rest sound small but genuinely help more than people expect.',
      "If this feeling is sticking around for weeks rather than passing, talking to someone about it is worth doing.",
    ],
  },
  surprised: {
    headline: 'Something caught you off guard.',
    tips: [
      "Give yourself a beat before reacting — the surprise wears off fast, and clearer thinking follows right behind it.",
    ],
  },
  calm: {
    headline: 'Nice and settled.',
    tips: [
      'A good state for making decisions or having a hard conversation, if either is on your plate.',
    ],
  },
};

export function getAdvice(emotion) {
  return EMOTION_ADVICE[emotion] || null;
}
