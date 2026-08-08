// Handles both entry points from Home: source === 'live' (mic recording)
// or source === 'upload' (file picker). Either way it ends by calling the
// backend and pushing the Result screen.
import React, { useEffect, useRef, useState } from 'react';
import { StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { Audio } from 'expo-av';
import * as DocumentPicker from 'expo-document-picker';
import * as Haptics from 'expo-haptics';

import Orb from '../components/Orb';
import WaveformVisualizer from '../components/WaveformVisualizer';
import { colors, spacing, typography } from '../theme';
import { predictEmotion } from '../services/api';
import { saveAnalysis } from '../services/storage';

const STATE_COPY = {
  idle: 'Tap the mic to start',
  recording: 'Listening…',
  analyzing: 'Analyzing tone, pitch & energy…',
  result: 'Done',
};

export default function RecordingScreen({ route, navigation }) {
  const source = route.params?.source || 'live';
  const [state, setState] = useState('idle');
  const [amplitude, setAmplitude] = useState(0);
  const [errorMsg, setErrorMsg] = useState(null);
  const recordingRef = useRef(null);
  const startedUploadRef = useRef(false);

  useEffect(() => {
    return () => {
      // best-effort cleanup if the user navigates away mid-recording
      if (recordingRef.current) {
        recordingRef.current.stopAndUnloadAsync().catch(() => {});
      }
    };
  }, []);

  useEffect(() => {
    if (source === 'upload' && !startedUploadRef.current) {
      startedUploadRef.current = true;
      handleUpload();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [source]);

  async function handleUpload() {
    try {
      const res = await DocumentPicker.getDocumentAsync({ type: 'audio/*', copyToCacheDirectory: true });
      if (res.canceled) {
        navigation.goBack();
        return;
      }
      const file = res.assets[0];
      setState('analyzing');
      await runInference({ uri: file.uri, name: file.name, mimeType: file.mimeType });
    } catch (e) {
      setErrorMsg(e.message || 'Could not read that file.');
      setState('idle');
    }
  }

  async function startRecording() {
    try {
      const perm = await Audio.requestPermissionsAsync();
      if (!perm.granted) {
        setErrorMsg('Microphone permission is required to record.');
        return;
      }
      await Audio.setAudioModeAsync({ allowsRecordingIOS: true, playsInSilentModeIOS: true });

      const { recording } = await Audio.Recording.createAsync(
        Audio.RecordingOptionsPresets.HIGH_QUALITY,
        (status) => {
          if (status.isRecording && typeof status.metering === 'number') {
            // metering is dBFS, roughly -160 (silence) .. 0 (max); normalize to 0..1
            const normalized = Math.max(0, Math.min(1, (status.metering + 60) / 60));
            setAmplitude(normalized);
          }
        },
        100
      );
      recordingRef.current = recording;
      setState('recording');
      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium).catch(() => {});
    } catch (e) {
      setErrorMsg('Could not start recording: ' + e.message);
    }
  }

  async function stopRecording() {
    const recording = recordingRef.current;
    if (!recording) return;
    try {
      setState('analyzing');
      await recording.stopAndUnloadAsync();
      const uri = recording.getURI();
      recordingRef.current = null;
      await runInference({ uri, name: 'recording.m4a', mimeType: 'audio/m4a' });
    } catch (e) {
      setErrorMsg('Could not process the recording: ' + e.message);
      setState('idle');
    }
  }

  async function runInference(file) {
    try {
      const result = await predictEmotion(file);
      setState('result');
      const saved = await saveAnalysis({
        emotion: result.emotion,
        confidence: result.confidence,
        probabilities: result.probabilities,
        modelUsed: result.model_used,
        durationMs: result.duration_ms,
        source,
        audioUri: file.uri,
      });
      navigation.replace('Result', { record: saved });
    } catch (e) {
      setErrorMsg(e.message || 'Inference failed.');
      setState('idle');
    }
  }

  const resultColor = null; // Result screen owns the final color; keep orb neutral here

  return (
    <SafeAreaView style={styles.safe}>
      <View style={styles.topBar}>
        <TouchableOpacity onPress={() => navigation.goBack()} hitSlop={12}>
          <Ionicons name="chevron-back" size={26} color={colors.textPrimary} />
        </TouchableOpacity>
      </View>

      <View style={styles.center}>
        <Orb state={state} amplitude={amplitude} emotionColor={resultColor} />
        <Text style={styles.stateText}>{STATE_COPY[state]}</Text>
        {errorMsg ? <Text style={styles.errorText}>{errorMsg}</Text> : null}

        <View style={styles.waveformWrap}>
          <WaveformVisualizer amplitude={amplitude} active={state === 'recording'} />
        </View>
      </View>

      <View style={styles.bottomBar}>
        {source === 'live' && state === 'idle' && (
          <TouchableOpacity style={styles.micButton} onPress={startRecording}>
            <Ionicons name="mic" size={30} color="#0B0D12" />
          </TouchableOpacity>
        )}
        {source === 'live' && state === 'recording' && (
          <TouchableOpacity style={[styles.micButton, styles.micButtonStop]} onPress={stopRecording}>
            <Ionicons name="stop" size={28} color="#F3F5F8" />
          </TouchableOpacity>
        )}
        {(state === 'analyzing') && <Text style={typography.caption}>Please hold on…</Text>}
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: colors.bg, justifyContent: 'space-between' },
  topBar: { paddingHorizontal: spacing.lg, paddingTop: spacing.sm },
  center: { alignItems: 'center', justifyContent: 'center', flex: 1, gap: spacing.lg },
  stateText: { ...typography.body, fontSize: 15 },
  errorText: { color: colors.danger, fontSize: 13, textAlign: 'center', paddingHorizontal: spacing.xl },
  waveformWrap: { height: 64, width: '100%' },
  bottomBar: { alignItems: 'center', paddingBottom: spacing.xxl, gap: spacing.sm, minHeight: 110, justifyContent: 'center' },
  micButton: {
    width: 76,
    height: 76,
    borderRadius: 38,
    backgroundColor: colors.accent,
    alignItems: 'center',
    justifyContent: 'center',
  },
  micButtonStop: {
    backgroundColor: colors.danger,
  },
});
