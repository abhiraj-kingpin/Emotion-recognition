// First thing anyone sees when the app opens. Auto-advances to Home after a
// beat, or immediately on tap - never traps someone on a splash screen.
import React, { useEffect } from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import Animated, {
  Easing,
  useAnimatedStyle,
  useSharedValue,
  withDelay,
  withTiming,
} from 'react-native-reanimated';

import Orb from '../components/Orb';
import { colors, spacing } from '../theme';

const AUTO_ADVANCE_MS = 2200;

export default function IntroScreen({ navigation }) {
  const orbOpacity = useSharedValue(0);
  const orbScale = useSharedValue(0.85);
  const titleOpacity = useSharedValue(0);
  const titleY = useSharedValue(10);
  const taglineOpacity = useSharedValue(0);

  useEffect(() => {
    orbOpacity.value = withTiming(1, { duration: 600, easing: Easing.out(Easing.ease) });
    orbScale.value = withTiming(1, { duration: 700, easing: Easing.out(Easing.back(1.2)) });
    titleOpacity.value = withDelay(300, withTiming(1, { duration: 500 }));
    titleY.value = withDelay(300, withTiming(0, { duration: 500, easing: Easing.out(Easing.ease) }));
    taglineOpacity.value = withDelay(600, withTiming(1, { duration: 500 }));

    const timer = setTimeout(goToHome, AUTO_ADVANCE_MS);
    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function goToHome() {
    navigation.replace('Home');
  }

  const orbStyle = useAnimatedStyle(() => ({
    opacity: orbOpacity.value,
    transform: [{ scale: orbScale.value }],
  }));
  const titleStyle = useAnimatedStyle(() => ({
    opacity: titleOpacity.value,
    transform: [{ translateY: titleY.value }],
  }));
  const taglineStyle = useAnimatedStyle(() => ({ opacity: taglineOpacity.value }));

  return (
    <Pressable style={styles.flex} onPress={goToHome}>
      <SafeAreaView style={styles.safe}>
        <View style={styles.center}>
          <Animated.View style={orbStyle}>
            <Orb state="idle" size={140} />
          </Animated.View>
          <Animated.Text style={[styles.title, titleStyle]}>EmotionSense</Animated.Text>
          <Animated.Text style={[styles.tagline, taglineStyle]}>
            Hear what the words don't say
          </Animated.Text>
        </View>
        <Animated.Text style={[styles.skipHint, taglineStyle]}>Tap to skip</Animated.Text>
      </SafeAreaView>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  flex: { flex: 1, backgroundColor: colors.bg },
  safe: { flex: 1, alignItems: 'center', justifyContent: 'center' },
  center: { alignItems: 'center', gap: spacing.md },
  title: {
    fontSize: 28,
    fontWeight: '700',
    color: colors.textPrimary,
    letterSpacing: 0.3,
  },
  tagline: {
    fontSize: 14,
    color: colors.textSecondary,
    marginTop: -spacing.sm,
  },
  skipHint: {
    position: 'absolute',
    bottom: spacing.xl,
    fontSize: 12,
    color: colors.textMuted,
  },
});
