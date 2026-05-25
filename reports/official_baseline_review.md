# 공식 베이스라인 노트북 검토

검토 파일:

```text
[Baseline]_Sequence LSTM + MLP 기반 CTR 예측.ipynb
```

## 구조 요약

- 모델: 비시퀀스 피처 MLP + `seq` LSTM 결합 모델
- 데이터:
  - `train.parquet` 전체 로드
  - `test.parquet` 로드 후 `ID` 제거
  - `clicked=1` 전체 사용
  - `clicked=0`은 양성 개수의 2배만 무작위 다운샘플링
- 피처:
  - `clicked`, `seq`, `ID`를 제외한 모든 컬럼을 tabular feature로 사용
  - tabular feature는 전부 float 변환 후 결측 0 대체
  - `seq`는 comma-separated 숫자 시퀀스를 매 batch에서 파싱
- 검증:
  - 다운샘플된 train에서 `train_test_split(test_size=0.2, shuffle=True)`
  - day7 holdout 사용 안 함
- 학습:
  - `BCEWithLogitsLoss`
  - Adam, learning rate 1e-3
  - batch size 4096
  - epoch 10
- 추론:
  - sigmoid 출력값을 그대로 제출

## 우리 파이프라인과의 차이

| 항목 | 공식 베이스라인 | 현재 파이프라인 |
|---|---|---|
| 주 모델 | LSTM + MLP | LightGBM + 블렌딩 |
| `seq` 처리 | 전체 시퀀스를 LSTM에 입력 | 길이/처음/마지막/고유 토큰 등 요약 피처 |
| 불균형 처리 | 음성 2배 다운샘플링 | class-balanced sample weight |
| 검증 | random split | day7 holdout 중심 |
| 확률 스케일 | 다운샘플 posterior | class-balanced raw prediction |
| 앙상블 | 없음 | baseline/seq/stable count 블렌드 |

## 가져올 만한 점

1. `seq` 전체를 모델링하는 별도 모델을 만든다.
   - 현재 `seq` 요약 피처만으로도 블렌딩 효과가 있었으므로, LSTM/GRU/1D CNN/hash 모델을 별도 예측기로 만들면 앙상블 다양성이 생길 가능성이 있다.

2. 다운샘플링 기반 학습을 별도 모델로 실험한다.
   - 현재 class-balanced weight와 비슷한 목적이지만 학습 데이터 분포가 달라져 예측 rank가 달라질 수 있다.
   - 단독 점수보다 기존 최고 블렌드에 5~20% 섞는 후보로 보는 것이 안전하다.

3. deep model 검증도 day7로 바꾼다.
   - 공식 노트북의 random split은 test가 day7-only인 상황과 맞지 않는다.
   - 그대로 따라 하면 로컬 검증이 지나치게 낙관적일 수 있다.

## 바로 쓰기 어려운 점

- `train.parquet` 전체를 pandas로 로드하므로 메모리 부담이 크다.
- `seq` 전체를 batch마다 `np.fromstring`으로 파싱해서 학습 속도가 느릴 수 있다.
- random validation은 현재 대회 구조와 맞지 않는다.
- downsampling 비율 1:2가 public/private에 최적인지는 알 수 없다.

## 다음 실험 제안

현재 최고 모델은 다음과 같다.

```text
submissions/blend_best80_stable_count20.csv
Public 0.3408619882
```

다음 실험은 공식 베이스라인을 그대로 제출하기보다, 구조만 차용한 별도 `seq` 모델을 만든 뒤 낮은 비율로 블렌딩하는 방향이 좋다.

우선순위:

1. `blend_best70_stable_count30.csv` 제출로 stable count 최적 비중 확인
2. `seq` 전용 경량 모델 생성
   - 후보: hashed sequence logistic/MLP, GRU/LSTM, sequence token embedding pooling
   - 검증: day7 holdout
   - 제출: 단독보다 기존 최고와 5~20% 블렌드 우선
3. deep sequence 모델을 만들 때는 전체 train pandas 로드 대신 sample/streaming 또는 parquet column select를 사용
