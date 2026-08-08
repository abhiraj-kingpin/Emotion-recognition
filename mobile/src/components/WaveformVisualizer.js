// Simple bar-style waveform driven by a rolling amplitude history (0..1 values).
// Used on the Recording screen underneath the orb while `state === 'recording'`.
import React, { useEffect, useRef, useState } from 'react';
import { StyleSheet, View } from 'react-native';
import { colors, radius } from '../theme';

const BAR_COUNT = 28;

export default function WaveformVisualizer({ amplitude = 0, active = false, color = colors.accent }) {
  const [levels, setLevels] = useState(() => new Array(BAR_COUNT).fill(0.06));
  const idx = useRef(0);

  useEffect(() => {
    if (!active) {
      setLevels(new Array(BAR_COUNT).fill(0.06));
      idx.current = 0;
      return;
    }
    setLevels((prev) => {
      const next = prev.slice(1);
      next.push(Math.max(0.06, Math.min(1, amplitude)));
      return next;
    });
  }, [amplitude, active]);

  return (
    <View style={styles.row}>
      {levels.map((lvl, i) => (
        <View
          key={i}
          style={[
            styles.bar,
            {
              height: 6 + lvl * 54,
              backgroundColor: color,
              opacity: 0.35 + lvl * 0.65,
            },
          ]}
        />
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  row: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    justifyContent: 'center',
    height: 64,
    gap: 3,
  },
  bar: {
    width: 3,
    borderRadius: radius.sm,
  },
});
