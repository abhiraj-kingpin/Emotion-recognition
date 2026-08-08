import React, { useCallback, useMemo, useState } from 'react';
import { FlatList, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useFocusEffect } from '@react-navigation/native';

import EmotionIcon from '../components/EmotionIcon';
import { colors, radius, spacing, typography } from '../theme';
import { EMOTIONS, emotionColor, emotionLabel } from '../constants/emotions';
import { getHistory } from '../services/storage';

export default function HistoryScreen({ navigation }) {
  const [history, setHistory] = useState([]);
  const [filter, setFilter] = useState(null);

  useFocusEffect(
    useCallback(() => {
      getHistory().then(setHistory);
    }, [])
  );

  const filtered = useMemo(
    () => (filter ? history.filter((h) => h.emotion === filter) : history),
    [history, filter]
  );

  return (
    <SafeAreaView style={styles.safe} edges={['top']}>
      <View style={styles.header}>
        <Text style={typography.h2}>History</Text>
        <Text style={typography.body}>{history.length} analyses saved on this device</Text>
      </View>

      <FlatList
        horizontal
        showsHorizontalScrollIndicator={false}
        data={[null, ...EMOTIONS]}
        keyExtractor={(item) => item || 'all'}
        style={styles.filterRow}
        contentContainerStyle={{ paddingHorizontal: spacing.lg, gap: spacing.sm }}
        renderItem={({ item }) => {
          const active = filter === item;
          const color = item ? emotionColor(item) : colors.accent;
          return (
            <TouchableOpacity
              style={[
                styles.chip,
                { borderColor: active ? color : colors.cardBorder, backgroundColor: active ? `${color}22` : colors.card },
              ]}
              onPress={() => setFilter(item)}
            >
              <Text style={[styles.chipText, active && { color }]}>{item ? emotionLabel(item) : 'All'}</Text>
            </TouchableOpacity>
          );
        }}
      />

      <FlatList
        data={filtered}
        keyExtractor={(item) => item.id}
        contentContainerStyle={styles.listContent}
        ListEmptyComponent={
          <View style={styles.empty}>
            <Ionicons name="albums-outline" size={32} color={colors.textMuted} />
            <Text style={styles.emptyText}>No analyses match this filter yet.</Text>
          </View>
        }
        renderItem={({ item }) => (
          <TouchableOpacity
            style={styles.row}
            onPress={() => navigation.navigate('Result', { record: item, fromHistory: true })}
          >
            <EmotionIcon emotion={item.emotion} />
            <View style={styles.rowMid}>
              <Text style={styles.rowTitle}>{emotionLabel(item.emotion)}</Text>
              <Text style={styles.rowSubtitle}>{new Date(item.timestamp).toLocaleString()}</Text>
            </View>
            <Text style={styles.rowConfidence}>{Math.round(item.confidence * 100)}%</Text>
          </TouchableOpacity>
        )}
      />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: colors.bg },
  header: { paddingHorizontal: spacing.lg, paddingTop: spacing.sm, paddingBottom: spacing.md, gap: 4 },
  filterRow: { flexGrow: 0, marginBottom: spacing.sm },
  chip: {
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: radius.pill,
    borderWidth: 1,
  },
  chipText: { color: colors.textSecondary, fontSize: 13, fontWeight: '600' },
  listContent: { paddingHorizontal: spacing.lg, paddingBottom: spacing.xxl, gap: spacing.sm },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.md,
    backgroundColor: colors.card,
    borderWidth: 1,
    borderColor: colors.cardBorder,
    borderRadius: radius.md,
    padding: spacing.md,
  },
  rowMid: { flex: 1, gap: 2 },
  rowTitle: { color: colors.textPrimary, fontWeight: '600', fontSize: 15 },
  rowSubtitle: { color: colors.textMuted, fontSize: 12 },
  rowConfidence: { color: colors.textSecondary, fontWeight: '700', fontSize: 14 },
  empty: { alignItems: 'center', gap: spacing.sm, paddingTop: spacing.xxl },
  emptyText: { color: colors.textMuted, fontSize: 13 },
});
