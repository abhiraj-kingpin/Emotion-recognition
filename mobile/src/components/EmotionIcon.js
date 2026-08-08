import React from 'react';
import { StyleSheet, View } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { EMOTION_META } from '../constants/emotions';

export default function EmotionIcon({ emotion, size = 28, containerSize = 56 }) {
  const meta = EMOTION_META[emotion] || { color: '#9AA5B1', icon: 'help-circle-outline' };
  return (
    <View
      style={[
        styles.wrap,
        {
          width: containerSize,
          height: containerSize,
          borderRadius: containerSize / 2,
          backgroundColor: `${meta.color}26`,
          borderColor: `${meta.color}55`,
        },
      ]}
    >
      <Ionicons name={meta.icon} size={size} color={meta.color} />
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: {
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
  },
});
