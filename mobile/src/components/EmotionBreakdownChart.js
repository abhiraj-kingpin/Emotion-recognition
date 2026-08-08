import React from 'react';
import { StyleSheet, Text, View } from 'react-native';
import { colors, radius, spacing } from '../theme';
import { EMOTION_META, emotionLabel } from '../constants/emotions';

/**
 * @param {{ probabilities: Record<string, number> }} props
 */
export default function EmotionBreakdownChart({ probabilities = {} }) {
  const rows = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);
  return (
    <View style={styles.wrap}>
      {rows.map(([emotion, prob]) => {
        const color = EMOTION_META[emotion]?.color || colors.accent;
        const pct = Math.round(prob * 100);
        return (
          <View key={emotion} style={styles.row}>
            <Text style={styles.label}>{emotionLabel(emotion)}</Text>
            <View style={styles.track}>
              <View style={[styles.fill, { width: `${pct}%`, backgroundColor: color }]} />
            </View>
            <Text style={styles.pct}>{pct}%</Text>
          </View>
        );
      })}
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: { gap: spacing.sm },
  row: { flexDirection: 'row', alignItems: 'center', gap: spacing.sm },
  label: { width: 78, color: colors.textSecondary, fontSize: 13, fontWeight: '500' },
  track: {
    flex: 1,
    height: 10,
    borderRadius: radius.pill,
    backgroundColor: colors.bgElevated,
    overflow: 'hidden',
  },
  fill: { height: '100%', borderRadius: radius.pill },
  pct: { width: 40, textAlign: 'right', color: colors.textPrimary, fontSize: 13, fontWeight: '600' },
});
