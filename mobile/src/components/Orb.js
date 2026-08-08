// The signature visual of the Recording/Analysis screen.
// States: idle (soft breathing pulse) -> recording (amplitude-driven pulse,
// amplitude in [0,1] comes from expo-av metering) -> analyzing (spinning
// shimmer ring, "thinking") -> result (settles + recolors to the detected
// emotion).
import React, { useEffect } from 'react';
import { StyleSheet, View } from 'react-native';
import Svg, { Circle, Defs, RadialGradient, Stop } from 'react-native-svg';
import Animated, {
  Easing,
  cancelAnimation,
  useAnimatedProps,
  useSharedValue,
  withRepeat,
  withSequence,
  withTiming,
} from 'react-native-reanimated';
import { colors } from '../theme';

const AnimatedCircle = Animated.createAnimatedComponent(Circle);
const SIZE = 220;
const CENTER = SIZE / 2;

export default function Orb({ state = 'idle', amplitude = 0, emotionColor = null, size = SIZE }) {
  const scale = useSharedValue(1);
  const ringOpacity = useSharedValue(0.35);
  const ringScale = useSharedValue(1);
  const rotation = useSharedValue(0);
  const color = useSharedValue(colors.accent);

  const activeColor = emotionColor || colors.accent;

  useEffect(() => {
    color.value = withTiming(0, { duration: 1 }); // no-op placeholder for color driver below
  }, []);

  useEffect(() => {
    cancelAnimation(scale);
    cancelAnimation(ringOpacity);
    cancelAnimation(ringScale);
    cancelAnimation(rotation);

    if (state === 'idle') {
      scale.value = withRepeat(
        withSequence(
          withTiming(1.06, { duration: 1600, easing: Easing.inOut(Easing.ease) }),
          withTiming(1.0, { duration: 1600, easing: Easing.inOut(Easing.ease) })
        ),
        -1,
        false
      );
      ringOpacity.value = withRepeat(
        withSequence(
          withTiming(0.5, { duration: 1600 }),
          withTiming(0.15, { duration: 1600 })
        ),
        -1,
        false
      );
      ringScale.value = withRepeat(
        withSequence(
          withTiming(1.15, { duration: 1600 }),
          withTiming(1.0, { duration: 1600 })
        ),
        -1,
        false
      );
    } else if (state === 'recording') {
      // scale/ring driven imperatively by amplitude updates below; keep a gentle
      // continuous rotation on the outer ring for a "listening" feel.
      rotation.value = withRepeat(withTiming(360, { duration: 6000, easing: Easing.linear }), -1, false);
    } else if (state === 'analyzing') {
      rotation.value = withRepeat(withTiming(360, { duration: 1100, easing: Easing.linear }), -1, false);
      scale.value = withRepeat(
        withSequence(
          withTiming(1.04, { duration: 500 }),
          withTiming(0.98, { duration: 500 })
        ),
        -1,
        false
      );
      ringOpacity.value = withRepeat(
        withSequence(withTiming(0.7, { duration: 550 }), withTiming(0.25, { duration: 550 })),
        -1,
        false
      );
    } else if (state === 'result') {
      scale.value = withSequence(
        withTiming(1.18, { duration: 260, easing: Easing.out(Easing.back(2)) }),
        withTiming(1.0, { duration: 320, easing: Easing.out(Easing.ease) })
      );
      ringOpacity.value = withTiming(0.4, { duration: 500 });
      ringScale.value = withTiming(1.08, { duration: 500 });
    }
  }, [state]);

  useEffect(() => {
    if (state === 'recording') {
      const target = 1 + Math.min(amplitude, 1) * 0.35;
      scale.value = withTiming(target, { duration: 90, easing: Easing.out(Easing.ease) });
      ringOpacity.value = withTiming(0.25 + Math.min(amplitude, 1) * 0.5, { duration: 90 });
      ringScale.value = withTiming(1 + Math.min(amplitude, 1) * 0.45, { duration: 90 });
    }
  }, [amplitude, state]);

  const coreProps = useAnimatedProps(() => ({
    r: (size * 0.28) * scale.value,
  }));

  const ringProps = useAnimatedProps(() => ({
    r: (size * 0.36) * ringScale.value,
    opacity: ringOpacity.value,
  }));

  const ring2Props = useAnimatedProps(() => ({
    r: (size * 0.42) * ringScale.value,
    opacity: ringOpacity.value * 0.5,
  }));

  return (
    <View style={{ width: size, height: size, alignItems: 'center', justifyContent: 'center' }}>
      <Svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        <Defs>
          <RadialGradient id="orbGrad" cx="50%" cy="42%" r="65%">
            <Stop offset="0%" stopColor={activeColor} stopOpacity="1" />
            <Stop offset="100%" stopColor={activeColor} stopOpacity="0.35" />
          </RadialGradient>
        </Defs>
        <AnimatedCircle
          cx={CENTER * (size / SIZE)}
          cy={CENTER * (size / SIZE)}
          animatedProps={ring2Props}
          fill="none"
          stroke={activeColor}
          strokeWidth={1.5}
        />
        <AnimatedCircle
          cx={CENTER * (size / SIZE)}
          cy={CENTER * (size / SIZE)}
          animatedProps={ringProps}
          fill="none"
          stroke={activeColor}
          strokeWidth={2}
        />
        <AnimatedCircle
          cx={CENTER * (size / SIZE)}
          cy={CENTER * (size / SIZE)}
          animatedProps={coreProps}
          fill="url(#orbGrad)"
        />
      </Svg>
    </View>
  );
}

const styles = StyleSheet.create({});
