import React from 'react';
import { StyleSheet, TouchableOpacity, View } from 'react-native';
import { colors, radius } from '../theme';

export default function Card({ children, style, onPress, elevated = false }) {
  const content = (
    <View style={[styles.card, elevated && styles.elevated, style]}>{children}</View>
  );
  if (!onPress) return content;
  return (
    <TouchableOpacity activeOpacity={0.75} onPress={onPress}>
      {content}
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: colors.card,
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: colors.cardBorder,
    padding: 16,
  },
  elevated: {
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.35,
    shadowRadius: 16,
    elevation: 6,
  },
});
