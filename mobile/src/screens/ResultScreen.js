import React from 'react';
import { ScrollView, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';

import Card from '../components/Card';
import Orb from '../components/Orb';
import EmotionBreakdownChart from '../components/EmotionBreakdownChart';
import { colors, radius, spacing, typography } from '../theme';
import { emotionColor, emotionLabel } from '../constants/emotions';

export default function ResultScreen({ route, navigation }) {
  const { record } = route.params;
  const color = emotionColor(record.emotion);

  return (
    <SafeAreaView style={styles.safe}>
      <View style={styles.topBar}>
        <TouchableOpacity onPress={() => navigation.navigate('Home')} hitSlop={12}>
          <Ionicons name="close" size={26} color={colors.textPrimary} />
        </TouchableOpacity>
        <Text style={typography.h3}>Analysis Result</Text>
        <View style={{ width: 26 }} />
      </View>

      <ScrollView contentContainerStyle={styles.content}>
        <View style={styles.orbWrap}>
          <Orb state="result" emotionColor={color} size={160} />
        </View>

        <Text style={[styles.emotionLabel, { color }]}>{emotionLabel(record.emotion)}</Text>
        <Text style={styles.confidence}>{Math.round(record.confidence * 100)}% confidence</Text>

        <View style={styles.metaRow}>
          <View style={styles.metaItem}>
            <Ionicons name="time-outline" size={16} color={colors.textMuted} />
            <Text style={styles.metaText}>
              {record.durationMs ? `${(record.durationMs / 1000).toFixed(2)}s inference` : '—'}
            </Text>
          </View>
          <View style={styles.metaItem}>
            <Ionicons name="hardware-chip-outline" size={16} color={colors.textMuted} />
            <Text style={styles.metaText}>{(record.modelUsed || 'cnn').toUpperCase()}</Text>
          </View>
        </View>

        <Card style={styles.breakdownCard}>
          <Text style={styles.breakdownTitle}>Emotion Breakdown</Text>
          <EmotionBreakdownChart probabilities={record.probabilities || {}} />
        </Card>

        <View style={styles.actionsRow}>
          <TouchableOpacity
            style={styles.actionBtn}
            onPress={() => navigation.navigate('Recording', { source: record.source || 'live' })}
          >
            <Ionicons name="refresh" size={18} color={colors.textPrimary} />
            <Text style={styles.actionText}>Re-analyze</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[styles.actionBtn, styles.actionBtnPrimary]}
            onPress={() => navigation.navigate('History')}
          >
            <Ionicons name="albums-outline" size={18} color="#0B0D12" />
            <Text style={[styles.actionText, { color: '#0B0D12' }]}>View History</Text>
          </TouchableOpacity>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: colors.bg },
  topBar: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: spacing.lg,
    paddingTop: spacing.sm,
  },
  content: { alignItems: 'center', padding: spacing.lg, paddingBottom: spacing.xxl, gap: spacing.sm },
  orbWrap: { marginVertical: spacing.md },
  emotionLabel: { fontSize: 30, fontWeight: '700' },
  confidence: { color: colors.textSecondary, fontSize: 14, marginBottom: spacing.sm },
  metaRow: { flexDirection: 'row', gap: spacing.lg, marginBottom: spacing.lg },
  metaItem: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  metaText: { color: colors.textMuted, fontSize: 12 },
  breakdownCard: { width: '100%', gap: spacing.sm },
  breakdownTitle: { color: colors.textPrimary, fontWeight: '600', fontSize: 15, marginBottom: 4 },
  actionsRow: { flexDirection: 'row', gap: spacing.md, width: '100%', marginTop: spacing.lg },
  actionBtn: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    paddingVertical: 14,
    borderRadius: radius.pill,
    borderWidth: 1,
    borderColor: colors.cardBorder,
  },
  actionBtnPrimary: { backgroundColor: colors.accent, borderColor: colors.accent },
  actionText: { color: colors.textPrimary, fontWeight: '600', fontSize: 14 },
});
