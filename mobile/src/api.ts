// PC 서버(foodcam.server)와 주고받는 함수와 응답 형식
import { File } from 'expo-file-system';
import { ImageManipulator, SaveFormat } from 'expo-image-manipulator';

export type Bowl = {
  x: number;
  y: number;
  r: number;
  food: string | null;
  grams: number | null;
  kcal: number | null;
  rimMm: number | null;
  source: string | null; // 영양성분 출처: 식약처 / AI 추정 / meal.toml 순서
  hull: [number, number][];
};

export type FrameResult = {
  width: number; // 서버가 계산한 사진 크기 (좌표 기준)
  height: number;
  scale: number | null; // px/mm
  problem: string | null;
  coin: { x: number; y: number; r: number } | null;
  bowls: Bowl[];
  status: { text: string; state: 'ok' | 'over' | 'under' }[];
  advice: { action: 'less' | 'more' | 'ok'; text: string } | null;
  speak: string | null; // 지금 소리 내 읽을 문장 (서버가 타이밍을 정함)
  recognition: boolean; // 서버가 음식을 인식하는지 (false면 meal.toml 순서대로 이름을 붙임)
};

// 무료 클라우드 서버는 잠들었다 깨는 데 30~60초 걸린다. 그동안은 시간 초과가 나고 다음 사진으로 다시 시도한다.
const TIMEOUT_MS = 20000;
// 휴대폰 원본(1200만 화소 이상)을 그대로 보내면 서버에서만 5초가 걸린다. 긴 변 1280px면 0.5초 안쪽.
const MAX_SIDE = 1280;

async function post(url: string, body: FormData): Promise<Response> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), TIMEOUT_MS);
  try {
    const res = await fetch(url, { method: 'POST', body, signal: ctrl.signal });
    if (!res.ok) throw new Error(`HTTP ${res.status}: ${(await res.text()).slice(0, 120)}`);
    return res;
  } catch (e) {
    if (ctrl.signal.aborted) {
      throw new Error(`${TIMEOUT_MS / 1000}초 안에 응답이 없음 (서버가 깨어나는 중이면 1분쯤 기다려 주세요)`);
    }
    throw e;
  } finally {
    clearTimeout(timer);
  }
}

async function shrinkPhoto(uri: string): Promise<string> {
  const ref = await ImageManipulator.manipulate(uri, { maxWidth: MAX_SIDE, maxHeight: MAX_SIDE }).renderAsync();
  const out = await ref.saveAsync({ compress: 0.7, format: SaveFormat.JPEG });
  return out.uri;
}

export async function analyzeFrame(server: string, session: string, photoUri: string): Promise<FrameResult> {
  const form = new FormData();
  // SDK 58의 fetch(expo/fetch)는 예전 RN 방식 { uri, name, type }을 못 받는다 → Blob처럼 동작하는 File로 보낸다
  form.append('image', new File(await shrinkPhoto(photoUri)));
  form.append('session', session);
  const res = await post(`${server.replace(/\/$/, '')}/frame`, form);
  const text = await res.text();
  let data: Partial<FrameResult>;
  try {
    data = JSON.parse(text);
  } catch {
    throw new Error(`서버 응답이 JSON이 아님: ${text.slice(0, 150)}`);
  }
  if (!Array.isArray(data.bowls) || !Array.isArray(data.status)) {
    console.warn('예상과 다른 서버 응답', res.status, res.url, text.slice(0, 500));
    throw new Error(`예상과 다른 서버 응답 (${res.status} ${res.url}): ${text.slice(0, 150)}`);
  }
  return data as FrameResult;
}

export async function resetBowls(server: string, session: string): Promise<void> {
  const form = new FormData();
  form.append('session', session);
  await post(`${server.replace(/\/$/, '')}/reset`, form);
}
