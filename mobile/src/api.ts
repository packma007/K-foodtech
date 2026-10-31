// PC 서버(foodcam.server)와 주고받는 함수와 응답 형식
import { File } from 'expo-file-system';

export type Bowl = {
  x: number;
  y: number;
  r: number;
  food: string | null;
  grams: number | null;
  kcal: number | null;
  rimMm: number | null;
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
};

const TIMEOUT_MS = 8000;

async function post(url: string, body: FormData): Promise<Response> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), TIMEOUT_MS);
  try {
    const res = await fetch(url, { method: 'POST', body, signal: ctrl.signal });
    if (!res.ok) throw new Error(`HTTP ${res.status}: ${(await res.text()).slice(0, 120)}`);
    return res;
  } finally {
    clearTimeout(timer);
  }
}

export async function analyzeFrame(server: string, session: string, photoUri: string): Promise<FrameResult> {
  const form = new FormData();
  // SDK 58의 fetch(expo/fetch)는 예전 RN 방식 { uri, name, type }을 못 받는다 → Blob처럼 동작하는 File로 보낸다
  form.append('image', new File(photoUri));
  form.append('session', session);
  const res = await post(`${server.replace(/\/$/, '')}/frame`, form);
  return res.json();
}

export async function resetBowls(server: string, session: string): Promise<void> {
  const form = new FormData();
  form.append('session', session);
  await post(`${server.replace(/\/$/, '')}/reset`, form);
}
