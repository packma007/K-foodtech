// 배식 도우미: 카메라로 그릇을 찍어 PC 서버(foodcam.server)로 보내고,
// 돌아온 결과(그릇·무게·영양 합계·안내)를 화면에 겹쳐 그리고 안내를 소리로 읽는다.
import AsyncStorage from '@react-native-async-storage/async-storage';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { useKeepAwake } from 'expo-keep-awake';
import * as Speech from 'expo-speech';
import { StatusBar } from 'expo-status-bar';
import { useCallback, useEffect, useRef, useState } from 'react';
import {
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  useWindowDimensions,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import Svg, { Circle, Polygon } from 'react-native-svg';

import { analyzeFrame, FrameResult, resetBowls } from './src/api';

const DEFAULT_SERVER = 'https://foodcam-server.onrender.com';
const SERVER_KEY = 'serverUrl';
const SESSION = Math.random().toString(36).slice(2, 10); // 앱을 켤 때마다 새 세션

const STATE_COLORS: Record<string, string> = { ok: '#78e678', over: '#ff5a5a', under: '#ffd250' };

export default function App() {
  useKeepAwake();
  const [permission, requestPermission] = useCameraPermissions();
  const camera = useRef<CameraView>(null);
  const { width: screenW } = useWindowDimensions();

  const [server, setServer] = useState(DEFAULT_SERVER);
  const [running, setRunning] = useState(false);
  const [muted, setMuted] = useState(false);
  const [result, setResult] = useState<FrameResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [fps, setFps] = useState(0);
  const runningRef = useRef(false);
  const mutedRef = useRef(false);

  useEffect(() => {
    AsyncStorage.getItem(SERVER_KEY).then((v) => v && setServer(v));
  }, []);
  useEffect(() => {
    mutedRef.current = muted;
    if (muted) Speech.stop();
  }, [muted]);

  // 찍기 → 보내기 → 결과 받기를 멈출 때까지 반복
  const loop = useCallback(async () => {
    while (runningRef.current) {
      const started = Date.now();
      try {
        const pic = await camera.current?.takePictureAsync({ quality: 0.4, shutterSound: false, exif: true });
        if (!pic || !runningRef.current) break;
        const orientation = Number(pic.exif?.Orientation ?? pic.exif?.orientation) || undefined;
        const r = await analyzeFrame(server, SESSION, pic.uri, orientation);
        setResult(r);
        setError(null);
        if (r.speak && !mutedRef.current) {
          Speech.stop();
          Speech.speak(r.speak, { language: 'ko-KR' });
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
        await new Promise((ok) => setTimeout(ok, 1000));
      }
      setFps(1000 / Math.max(Date.now() - started, 1));
    }
  }, [server]);

  const toggle = () => {
    const next = !running;
    runningRef.current = next;
    setRunning(next);
    if (next) {
      AsyncStorage.setItem(SERVER_KEY, server);
      loop();
    }
  };

  if (!permission) return <View style={styles.root} />;
  if (!permission.granted) {
    return (
      <SafeAreaView style={[styles.root, styles.center]}>
        <Text style={styles.text}>그릇을 찍으려면 카메라 권한이 필요합니다.</Text>
        <Pressable style={styles.button} onPress={requestPermission}>
          <Text style={styles.buttonText}>권한 허용</Text>
        </Pressable>
      </SafeAreaView>
    );
  }

  // 사진 비율에 맞춰 미리보기 크기를 정한다 (결과 좌표를 그대로 겹치기 위해)
  const aspect = result ? result.height / result.width : 4 / 3;
  const viewW = screenW;
  const viewH = screenW * aspect;
  const k = result ? viewW / result.width : 1;

  return (
    <SafeAreaView style={styles.root} edges={['top', 'bottom']}>
      <StatusBar style="light" />
      <View style={{ width: viewW, height: viewH }}>
        <CameraView ref={camera} style={StyleSheet.absoluteFill} facing="back" ratio="4:3" animateShutter={false} />
        {result && (
          <>
            <Svg style={StyleSheet.absoluteFill} width={viewW} height={viewH}>
              {result.coin && (
                <Circle cx={result.coin.x * k} cy={result.coin.y * k} r={result.coin.r * k}
                        stroke="#ffd700" strokeWidth={3} fill="none" />
              )}
              {result.bowls.map((b, i) => (
                <Circle key={`c${i}`} cx={b.x * k} cy={b.y * k} r={b.r * k}
                        stroke={b.food ? '#ffffff' : '#888888'} strokeWidth={2} fill="none" />
              ))}
              {result.bowls.map((b, i) => b.hull.length > 2 && (
                <Polygon key={`h${i}`} points={b.hull.map(([x, y]) => `${x * k},${y * k}`).join(' ')}
                         stroke="#50dc50" strokeWidth={2} fill="rgba(80,220,80,0.15)" />
              ))}
            </Svg>
            {result.bowls.map((b, i) => (
              <Text key={`t${i}`} style={[styles.bowlLabel, { left: (b.x - b.r) * k, top: (b.y + b.r) * k + 2 }]}>
                {b.food ?? '설정에 없는 그릇'}
                {b.grams != null ? ` ${b.grams}g` : ''}
                {b.kcal != null ? ` · ${b.kcal}kcal` : ''}
                {b.source ? ` (${b.source})` : ''}
              </Text>
            ))}
          </>
        )}
      </View>

      <ScrollView style={styles.panel} contentContainerStyle={{ padding: 12, gap: 6 }}>
        {result?.status.map((s) => (
          <Text key={s.text} style={[styles.text, { color: STATE_COLORS[s.state] }]}>{s.text}</Text>
        ))}
        {result && !result.recognition && (
          <Text style={[styles.text, { color: '#ffb347' }]}>
            음식 인식 꺼짐: 서버에 API 키가 없어 음식 이름을 설정 순서대로 붙이고 있어요
          </Text>
        )}
        {result?.problem ? (
          <Text style={[styles.text, { color: '#bbb' }]}>{result.problem}</Text>
        ) : result?.advice ? (
          <Text style={[styles.text, styles.advice]}>▶ {result.advice.text}</Text>
        ) : null}
        {error && <Text style={[styles.text, { color: '#ff7a7a' }]}>서버 연결 실패: {error}</Text>}
        <Text style={styles.small}>
          {running ? `${fps.toFixed(1)}장/초` : '정지됨'}
          {result?.scale ? ` · 축척 ${result.scale}px/mm` : ''}
        </Text>

        <View style={styles.row}>
          <Pressable style={[styles.button, running && styles.stop]} onPress={toggle}>
            <Text style={styles.buttonText}>{running ? '정지' : '시작'}</Text>
          </Pressable>
          <Pressable style={styles.button} onPress={() => resetBowls(server, SESSION).catch((e) => setError(String(e)))}>
            <Text style={styles.buttonText}>빈 그릇 다시 찍기</Text>
          </Pressable>
          <Pressable style={styles.button} onPress={() => setMuted((m) => !m)}>
            <Text style={styles.buttonText}>{muted ? '소리 켜기' : '소리 끄기'}</Text>
          </Pressable>
        </View>
        <TextInput
          style={styles.input}
          value={server}
          onChangeText={setServer}
          editable={!running}
          autoCapitalize="none"
          autoCorrect={false}
          keyboardType="url"
          placeholder="http://PC주소:8000"
          placeholderTextColor="#777"
        />
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#111' },
  center: { alignItems: 'center', justifyContent: 'center', gap: 16, padding: 24 },
  panel: { flex: 1 },
  text: { color: '#fff', fontSize: 16 },
  small: { color: '#999', fontSize: 13 },
  advice: { fontSize: 18, fontWeight: '600' },
  bowlLabel: {
    position: 'absolute',
    color: '#fff',
    fontSize: 13,
    backgroundColor: 'rgba(0,0,0,0.6)',
    paddingHorizontal: 4,
  },
  row: { flexDirection: 'row', gap: 8, marginTop: 6, flexWrap: 'wrap' },
  button: { backgroundColor: '#2b6cb0', paddingVertical: 10, paddingHorizontal: 14, borderRadius: 8 },
  stop: { backgroundColor: '#c53030' },
  buttonText: { color: '#fff', fontSize: 15, fontWeight: '600' },
  input: { borderWidth: 1, borderColor: '#444', borderRadius: 8, color: '#fff', padding: 8, marginTop: 6 },
});
