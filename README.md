# foodcam — 카메라로 음식 영양성분 보기

웹캠 영상에서 음식과 포장 제품을 찾아, 각각의 영양성분을 영상 위에 실시간으로 띄웁니다.

## 영양정보 우선순위

| 순위 | 대상 | 경로 |
|---|---|---|
| 1 | **실제 판매 제품** (바코드가 보일 때) | 바코드 → 식품안전나라 `C005`(없으면 `I2570`) → 품목제조보고번호 → 식약처 식품영양성분DB **상용제품** |
| 1 | 〃 (식약처 DB에 없을 때) | 바코드 → Open Food Facts |
| 2 | 제품 정보가 없거나, 화면 인식으로만 찾은 음식 | 식약처 식품영양성분DB **품목대표** (제품명 → 식품유형 → 음식명 순으로 검색) |

화면에는 출처가 `[식약처 상용제품]`, `[Open Food Facts]`, `[식약처 품목대표]` 중 하나로 표시됩니다.

## 설치

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env   # 그다음 인증키 입력
```

### 인증키

- `MFDS_SERVICE_KEY`: [공공데이터포털 식품영양성분DB정보](https://www.data.go.kr/data/15127578/openapi.do)에서 활용신청 → **일반 인증키(Decoding)**. 없으면 식약처 영양성분 조회를 건너뛰고 Open Food Facts만 씁니다.
- `FOODSAFETY_KEY`: [식품안전나라 OpenAPI](https://www.foodsafetykorea.go.kr/api/main.do) 인증키. 비워 두면 `sample` 키를 쓰는데, 개발용으로는 충분하지만 조회 건수가 제한됩니다.

## 실행

```bash
python -m foodcam.main                    # 웹캠
python -m foodcam.main --image 사진.jpg    # 사진 한 장
python -m foodcam.main --no-yolo          # 바코드만
python -m foodcam.main --model best.pt    # 직접 학습한 음식 모델

python -m foodcam.lookup 8801043014809 김치찌개   # 카메라 없이 조회만
```

## 배식 안내 (양 추정 + 음성)

그릇에 담긴 양을 실시간으로 추정해서, 미리 적어 둔 목표 영양성분에 맞게 "덜어 주세요 / 더 담아 주세요"를 음성으로 안내합니다.

```bash
copy meal.example.toml meal.toml        # 목표 영양성분·그릇·음식 적기
python -m foodcam.serving --meal meal.toml
python -m foodcam.serving --meal meal.toml --bowls hough   # 그릇을 YOLO 대신 원 검출로 찾기
```

**준비**
- 카메라를 상판 위에서 **수직으로** 내려다보게 둡니다. `camera_height_mm`에는 렌즈부터 상판까지의 거리를 적습니다.
- **500원 동전**을 그릇 옆 상판에 둡니다.
- **빈 그릇을 먼저** 보여 줍니다. 그 모습을 기준 영상으로 삼아 음식을 찾습니다. 그릇을 바꿨으면 `r` 키를 누릅니다.
- `meal.toml`의 `[[bowls]]`는 **화면 왼쪽 그릇부터** 순서대로 대응합니다.

**계산 방식** (자세한 식은 [foodcam/portion.py](foodcam/portion.py) 맨 위 설명)
1. 동전 지름(px) ÷ 26.5mm로 축척을 구합니다.
2. 그릇 윗면은 상판보다 카메라에 가까워서 더 크게 보이므로, 높이만큼 축척을 보정합니다.
3. 그릇 지름(mm)을 잽니다.
4. 빈 그릇 기준 영상과 달라진 부분을 음식 표면으로 보고, 그 넓이를 같은 넓이의 원(지름)으로 바꿉니다.
5. 그릇을 원뿔대로 보고 채운 높이와 부피를 구합니다.
6. 무게 = 부피 × 밀도, 영양성분 = 무게 × 100g당 함량 ÷ 100

**안내 규칙** ([foodcam/guide.py](foodcam/guide.py))
- 상한을 넘은 영양소가 있으면, 그 영양소를 가장 많이 낸 그릇에서 덜라고 안내합니다. 몇 g을 덜지도 계산합니다.
- 하한에 모자라면, 더 담아도 다른 상한을 넘지 않는 그릇 가운데 g당 함량이 높은 그릇에 더 담으라고 안내합니다.
- 상한 초과를 하한 부족보다 먼저 안내합니다.
- 판단이 1.2초 동안 그대로일 때만 말하고, 같은 안내는 8초 뒤에 다시 말합니다.

**합성 영상 시험 결과** (정답 무게를 알고 그린 장면): 밥 170g → 165g, 찌개 532g → 554g, 186g → 194g (오차 ±5% 안팎). 실제 카메라로는 아직 시험하지 않았습니다.

## 모바일 앱 + 서버

안드로이드 앱([mobile/](mobile/), Expo)이 사진을 서버로 보내면, 서버가 계산하고 앱이 화면 표시와 음성 안내를 합니다.

```bash
python -m foodcam.server --meal meal.toml   # PC에서 돌릴 때 (휴대폰과 같은 와이파이)
cd mobile && npx expo start --lan           # Expo Go로 시험
```

APK 빌드: `mobile/`에서 `EAS_NO_VCS=1 EAS_PROJECT_ROOT="$(pwd)" npx eas-cli build -p android --profile preview`

### 클라우드 서버 (Render)

[render.yaml](render.yaml)에 설정이 들어 있습니다.
- 설치 목록은 [requirements-server.txt](requirements-server.txt)를 씁니다. torch가 빠져 있어서 무료 플랜(512MB)에서도 돌아가고, 그릇은 원 검출(hough)로만 찾습니다.
- Render에서 **New → Blueprint**로 이 저장소를 고르면 됩니다.
- 식약처 키는 Render 환경 변수 `MFDS_SERVICE_KEY`에 넣습니다.
- 앱의 서버 주소 칸에 `https://<서비스 이름>.onrender.com`을 입력합니다.
- 무료 플랜은 15분 동안 안 쓰면 잠들어서, 첫 요청에 30~60초가 걸립니다.

## 구조

```
foodcam/
  main.py        카메라 루프 (조회는 백그라운드라 영상이 멈추지 않음)
  detectors.py   바코드 읽기(OpenCV) + 음식 탐지(YOLO)
  nutrition.py   영양성분 조회와 우선순위, 캐시
  overlay.py     박스와 영양성분 패널 그리기 (한글은 Pillow로)
  lookup.py      조회만 확인하는 CLI
  serving.py     배식 안내 실행 파일
  portion.py     동전 축척, 그릇 찾기·추적, 음식 표면, 부피 계산
  guide.py       목표와 비교해 덜기/더 담기 판단
  voice.py       한국어 음성 (Windows Heami, 오프라인)
  meal.py        meal.toml 읽기
```

## 한계와 다음 단계

- 기본 YOLO 모델(COCO)이 아는 음식은 바나나·사과·피자 등 10종뿐입니다. 한식을 인식하려면 [AI Hub 「음식 이미지 및 영양정보 텍스트」](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=74)(400종 이상)로 YOLO를 학습시킨 뒤 `--model`로 넘기면 됩니다. 이때 **클래스 이름을 한글 음식명으로** 두면 그 이름으로 식약처 DB를 검색합니다.
- 바코드가 안 보이는 포장 제품은 어느 회사 제품인지 알 수 없어서 품목대표값으로 대신합니다. 포장 글자를 읽는 OCR이나 비전 LLM을 붙이면 바코드 없이도 제품을 특정할 수 있습니다.
- 품목대표·상용제품 값은 대부분 100g(또는 100mL) 기준입니다. 실제 먹는 양은 반영하지 않습니다.
