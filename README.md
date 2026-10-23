# firstCV

무선으로 연결한 아이폰 카메라를 들고 돌아다니면서, 화면에 들어오는 **일상 사물을 실시간으로 인식**하는 객체 탐지 프로젝트.

**목표**: 데이터가 하나도 없는 상태에서 시작해 수집 → 라벨링 → 학습 → 개선 → 온디바이스 배포까지
객체 탐지 파이프라인 전체를 직접 돌려본다. 정확도(mAP)와 속도(FPS)는 항상 함께 잰다.

**요구사항**

- 카메라가 계속 움직인다 → 모션 블러, 다양한 거리·각도에서도 버텨야 한다
- 들고 다니며 끊김 없이 볼 수 있을 만큼 빨라야 한다
- 최종 배포는 아이폰 온디바이스(CoreML)

## 로드맵

- [ ] **1. 기성 모델 써보기** — 사전학습 YOLO를 아이폰 카메라에 돌려보고, 무엇을 잘 잡고 무엇을 놓치는지 관찰 ([기록](notes/01-pretrained-yolo.md))
- [ ] **2. 손으로 작게 한 바퀴** — 직접 찍고 100~200장 라벨링 → YOLO 파인튜닝 → mAP·FPS 측정
- [ ] **3. 자동 라벨링** — SAM 3.1 텍스트 프롬프트로 라벨 초안 → 검수만 → 데이터 10배
- [ ] **4. 극한으로 밀기** — 틀린 프레임 모아 재라벨링(액티브 러닝), 블러·조명 증강, 공개 데이터셋 섞기
- [ ] **5. 경량화·실시간** — CoreML 변환, 아이폰에서 실시간 인식
- [ ] **6. 결과 정리** — 단계별 mAP·FPS 그래프, 데모 영상, 회고

## 시작하기

```bash
cd ~/Developer/firstCV
uv sync
uv run scripts/01_webcam_yolo.py --list-cameras   # 카메라 번호 확인
uv run scripts/01_webcam_yolo.py --camera 1       # 아이폰(연속성 카메라)으로 탐지
```

- 처음 실행하면 macOS가 터미널(또는 VS Code)의 카메라 접근 권한을 물어본다. 허용하고 다시 실행한다.
- 아이폰은 맥과 같은 Apple ID로 로그인돼 있으면 연속성 카메라로 무선 연결된다. 보통 1번, 맥 내장 카메라는 0번.
- 모델 가중치(`*.pt`)는 처음 실행할 때 자동으로 내려받는다.

## 스크립트

| 파일 | 하는 일 |
|---|---|
| `scripts/01_webcam_yolo.py` | 카메라에 사전학습 YOLO 실행, 화면에 FPS 표시. `s` 원본 프레임 저장, `q` 종료 |
| `scripts/02_extract_frames.py` | 폰 영상에서 일정 간격으로 프레임 추출, 흐린 프레임은 자동으로 버림 |
| `scripts/03_train.py` | 공개 데이터셋(기본 HomeObjects-3K)으로 사전학습 YOLO 파인튜닝. 결과는 `runs/<이름>/weights/best.pt` |

자주 쓰는 옵션:

```bash
uv run scripts/01_webcam_yolo.py --camera 1 --model yolo11s.pt --conf 0.4   # 모델·임계값 바꾸기
uv run scripts/01_webcam_yolo.py --camera 1 --no-detect                     # 탐지 없이 촬영만 (데이터 수집)
uv run scripts/02_extract_frames.py data/videos/walk1.mov --every 0.5        # 0.5초마다 한 장
uv run scripts/02_extract_frames.py data/videos/walk1.mov --min-sharpness 0  # 블러 필터 끄기
uv run scripts/03_train.py --epochs 3 --fraction 0.1                         # 학습 빠르게 한 바퀴
uv run scripts/01_webcam_yolo.py --camera 1 --model runs/home_n/weights/best.pt  # 학습한 모델로 확인
```

## 비교해볼 모델

| 가중치 | 특징 |
|---|---|
| `yolo11n.pt` / `yolo11s.pt` | COCO 80클래스. n은 빠르고 s는 더 정확 |
| `yolo26n.pt` | 최신 YOLO26 nano |
| `yolov8n-oiv7.pt` | Open Images V7 600클래스. 일상 사물 범위가 훨씬 넓다 |
| `yoloe-26s-seg-pf.pt` | YOLOE 프롬프트 프리 모델. 클래스 목록 없이 보이는 걸 찾는 오픈 보캐뷸러리 |

## 데이터

| 위치 | 내용 |
|---|---|
| `data/videos/` | 아이폰으로 들고 다니며 찍은 원본 영상 |
| `data/raw/` | 촬영·추출한 원본 이미지 (라벨링 대상) |
| `data/datasets/homeobjects-3K/` | 공개 데이터셋 HomeObjects-3K. 실내 사물 12클래스, train 2,285장 / val 404장 |

`data/`의 내용물과 `*.pt`, `runs/`는 git에 올리지 않는다.

## 폴더

```
scripts/   단계별 실행 스크립트 (번호 순서대로)
notes/     단계별 관찰, 질문, 실험 기록
data/      영상·이미지·데이터셋 (git 제외)
```
