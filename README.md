# 토스 광고 클릭 예측(CTR) 모델 개발

DACON 토스 광고 클릭 예측 경진대회 실험용 작업 공간입니다.

이 저장소에는 첫 베이스라인 실험을 재현하기 위한 코드와 기록만 포함합니다.
원본 대회 데이터와 생성된 제출 파일은 용량이 크고 대회 데이터이므로 Git에는
올리지 않았습니다. 데이터는 DACON에서 직접 내려받아 프로젝트 루트에 배치해야
합니다.

## 대회 개요

- 과제: 광고 노출 로그 기반 클릭 여부 이진분류
- 타깃: `clicked`
- 평가: AP와 Weighted Log Loss
- 로컬에서 확인한 데이터 크기:
  - train: 10,704,179행, 119열
  - test: 1,527,298행, 119열
- 중요한 검증 힌트: test 전체가 `day_of_week = 7`

## 저장소 구성

| 경로 | 설명 |
|---|---|
| `metrics.py` | AP, Weighted Log Loss, 로컬 비교용 보조 점수 |
| `eda_summary.py` | parquet 스키마와 주요 분포 요약 리포트 생성 |
| `train_baseline.py` | day7/무작위 검증 기반 LightGBM 베이스라인 학습 |
| `make_submission.py` | 저장된 모델로 DACON 제출 파일 생성 |
| `calibrate_submission.py` | 사전확률 보정 실험 기록용 스크립트 |
| `blend_submissions.py` | 제출 파일 단순 블렌딩 유틸리티 |
| `reports/eda_summary.md` | 로컬 데이터 기준 EDA 요약 |
| `reports/experiment_log.md` | 제출 결과와 실험 기록 |

## 환경 준비

```bash
python3 -m pip install -r requirements.txt
```

DACON에서 받은 파일을 프로젝트 루트에 둡니다.

```text
train.parquet
test.parquet
sample_submission.csv
```

## 현재 베이스라인 재현

EDA 요약 생성:

```bash
python3 eda_summary.py --output reports/eda_summary.md
```

첫 day7 validation 베이스라인 학습:

```bash
python3 train_baseline.py \
  --output-dir models/baseline_lgbm \
  --train-sample-frac 0.10 \
  --valid-sample-frac 0.30 \
  --seed 42
```

제출 파일 생성:

```bash
python3 make_submission.py \
  --model-dir models/baseline_lgbm \
  --output submissions/baseline_lgbm.csv
```

기본 `seq` 파생 피처를 포함한 모델 학습:

```bash
python3 train_baseline.py \
  --output-dir models/baseline_lgbm_seq \
  --train-sample-frac 0.10 \
  --valid-sample-frac 0.30 \
  --use-seq-features \
  --seed 42
```

기본 `seq` 피처와 빈도 인코딩을 포함한 모델 학습:

```bash
python3 train_baseline.py \
  --output-dir models/baseline_lgbm_seq_count \
  --train-sample-frac 0.10 \
  --valid-sample-frac 0.30 \
  --use-seq-features \
  --use-count-features \
  --count-feature-set stable \
  --seed 42
```

## 현재 결과

| 제출 파일 | 로컬 검증 | 공개 리더보드 점수 | 메모 |
|---|---:|---:|---|
| `baseline_lgbm.csv` | AP 0.05833 / WLL 0.61761 | 0.340445203 | 첫 LightGBM 원본 예측 베이스라인 |
| `baseline_lgbm_prior_calibrated.csv` | n/a | 0.2045772218 | 실제 CTR 사전확률 보정은 점수를 크게 악화 |
| `baseline_lgbm_seq.csv` | AP 0.05890 / WLL 0.61836 | 미제출 | 기본 시퀀스 길이/처음/마지막/고유 토큰 피처 추가 |
| `blend_base80_seq20.csv` | n/a | 0.3405416183 | 베이스라인 80% + seq 20% 블렌드 |
| `blend_base70_seq30.csv` | n/a | 0.3405715349 | 베이스라인 70% + seq 30% 블렌드 |
| `blend_base60_seq40.csv` | n/a | 0.3405770109 | 베이스라인 60% + seq 40% 블렌드. 현재 최고 점수 |
| `blend_base55_seq45.csv` | n/a | 0.3405770109 | 베이스라인 55% + seq 45% 블렌드. 60:40과 동률 |
| `baseline_lgbm_seq_count.csv` | AP 0.06051 / WLL 0.61784 | 0.339500066 | 기본 seq 피처와 빈도 인코딩 추가. Public에서는 악화 |
| `blend_best90_seq_count10.csv` | n/a | 미제출 | 현재 최고 90% + seq_count 10% 저위험 블렌드 후보 |
| `baseline_lgbm_seq_stable_count.csv` | AP 0.06086 / WLL 0.61794 | 미제출 | 안정적인 count feature만 사용한 재설계 모델 |
| `blend_best90_stable_count10.csv` | n/a | 미제출 | 현재 최고 90% + stable_count 10% 블렌드 후보 |
| `blend_best80_stable_count20.csv` | n/a | 미제출 | 현재 최고 80% + stable_count 20% 블렌드 후보 |

## 현재까지의 결론

- test가 모두 `day_of_week = 7`이므로 day7 holdout을 주 검증셋으로 사용한다.
- 예측값을 실제 train CTR 사전확률까지 낮추는 보정은 사용하지 않는다. 공개 리더보드 점수가 크게 하락했다.
- 기본 `seq` 피처는 단독 제출보다 블렌딩에서 더 유용할 가능성이 있다.
- 80:20보다 70:30, 70:30보다 60:40이 좋았고, 55:45는 60:40과 동률이었다.
- 빈도 인코딩을 추가한 모델은 day7 검증 AP를 0.06051까지 올렸지만 Public 점수는 0.339500066으로 악화됐다.
- count encoding 재설계에서는 고카디널리티 조합과 중복적인 frequency 피처를 기본에서 제외했다.
- stable count 재설계 모델은 day7 검증 AP 0.06086을 기록했다.
- 다음으로는 `blend_best90_stable_count10.csv`처럼 낮은 비율 블렌드부터 조심스럽게 확인한다.
