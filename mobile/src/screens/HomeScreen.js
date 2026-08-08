import React, { useCallback, useState } from 'react';
import { ScrollView, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useFocusEffect } from '@react-navigation/native';
import Card from '../components/Card';
import EmotionIcon from '../components/EmotionIcon';
import { colors, radius, spacing, typography } from '../theme';
import { emotionLabel } from '../constants/emotions';
import { getHistory } from '../services/storage';

export default function HomeScreen({ navigation }) {
  const [recent, setRecent] = useState([]);

  useFocusEffect(
    useCallback(() => {
      getHistory().then((h) => setRecent(h.slice(0, 8)));
    }, [])
  );

  return (
    <SafeAreaView style={styles.safe} edges={['top']}>
      <ScrollView contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
        <View style={styles.header}>
          <Text style={typography.caption}>SPEECH EMOTION RECOGNITION</Text>
          <Text style={typography.h1}>How are they{'\n'}really feeling?</Text>
        </View>

        <TouchableOpacity
          style={styles.ctaPrimary}
          activeOpacity={0.85}
          onPress={() => navigation.navigate('Recording', { source: 'live' })}
        >
          <View>
            <Text style={styles.ctaTitle}>New Recording</Text>
            <Text style={styles.ctaSubtitle}>Analyze a live clip in seconds</Text>
          </View>
          <View style={styles.ctaIcon}>
            <Ionicons name="mic" size={26} color="#0B0D12" />
          </View>
        </TouchableOpacity>

        <View style={styles.row2}>
          <Card
            style={styles.halfCard}
            onPress={() => navigation.navigate('Recording', { source: 'live' })}
          >
            <Ionicons name="radio-outline" size={22} color={colors.accent} />
            <Text style={styles.cardTitle}>Live Record</Text>
            <Text style={styles.cardSubtitle}>Use your mic</Text>
          </Card>
          <Card
            style={styles.halfCard}
            onPress={() => navigation.navigate('Recording', { source: 'upload' })}
          >
            <Ionicons name="cloud-upload-outline" size={22} color={colors.accent} />
            <Text style={styles.cardTitle}>Upload Audio</Text>
            <Text style={styles.cardSubtitle}>From your files</Text>
          </Card>
        </View>

        <View style={styles.row2}>
          <Card style={styles.halfCard} onPress={() => navigation.navigate('History')}>
            <Ionicons name="time-outline" size={22} color={colors.success} />
            <Text style={styles.cardTitle}>View History</Text>
            <Text style={styles.cardSubtitle}>{recent.length} saved</Text>
          </Card>
          <Card style={styles.halfCard} onPress={() => navigation.navigate('History')}>
            <Ionicons name="stats-chart-outline" size={22} color={colors.success} />
            <Text style={styles.cardTitle}>Insights</Text>
            <Text style={styles.cardSubtitle}>Trends over time</Text>
          </Card>
        </View>

        <View style={styles.sectionHeader}>
          <Text style={typography.h3}>Recent Analyses</Text>
          <TouchableOpacity onPress={() => navigation.navigate('History')}>
            <Text style={styles.seeAll}>See all</Text>
          </TouchableOpacity>
        </View>

        {recent.length === 0 ? (
          <Card>
            <Text style={styles.emptyText}>
              No analyses yet — record or upload a clip to get started.
            </Text>
          </Card>
        ) : (
          <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.chipsRow}>
            {recent.map((item) => (
              <TouchableOpacity
                key={item.id}
                style={styles.chip}
                onPress={() => navigation.navigate('Result', { record: item, fromHistory: true })}
              >
                <EmotionIcon emotion={item.emotion} size={18} containerSize={34} />
                <Text style={styles.chipLabel}>{emotionLabel(item.emotion)}</Text>
                <Text style={styles.chipConfidence}>{Math.round(item.confidence * 100)}%</Text>
              </TouchableOpacity>
            ))}
          </ScrollView>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: colors.bg },
  content: { padding: spacing.lg, paddingBottom: spacing.xxl, gap: spacing.md },
  header: { marginBottom: spacing.sm, gap: spacing.xs },
  ctaPrimary: {
    backgroundColor: colors.accent,
    borderRadius: radius.lg,
    padding: spacing.lg,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  ctaTitle: { fontSize: 19, fontWeight: '700', color: '#0B0D12' },
  ctaSubtitle: { fontSize: 13, color: 'rgba(11,13,18,0.7)', marginTop: 2 },
  ctaIcon: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: 'rgba(11,13,18,0.15)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  row2: { flexDirection: 'row', gap: spacing.md },
  halfCard: { flex: 1, gap: 6 },
  cardTitle: { color: colors.textPrimary, fontWeight: '600', fontSize: 15, marginTop: 4 },
  cardSubtitle: { color: colors.textMuted, fontSize: 12 },
  sectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: spacing.sm,
  },
  seeAll: { color: colors.accent, fontSize: 13, fontWeight: '600' },
  emptyText: { color: colors.textMuted, fontSize: 13, lineHeight: 19 },
  chipsRow: { marginHorizontal: -spacing.lg, paddingHorizontal: spacing.lg },
  chip: {
    backgroundColor: colors.card,
    borderWidth: 1,
    borderColor: colors.cardBorder,
    borderRadius: radius.lg,
    padding: spacing.sm,
    marginRight: spacing.sm,
    alignItems: 'center',
    width: 92,
    gap: 4,
  },
  chipLabel: { color: colors.textPrimary, fontSize: 12, fontWeight: '600' },
  chipConfidence: { color: colors.textMuted, fontSize: 11 },
});
