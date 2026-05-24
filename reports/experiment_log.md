# 실험 기록

| 파일 | 로컬 검증 | 공개 리더보드 점수 | 메모 |
|---|---:|---:|---|
| `submissions/baseline_lgbm.csv` | AP 0.05833 / WLL 0.61761 | 0.340445203 | `seq`를 제외한 첫 LightGBM 원본 예측 베이스라인 |
| `submissions/baseline_lgbm_prior_calibrated.csv` | n/a | 0.2045772218 | 실제 CTR 사전확률 보정은 점수를 크게 악화. 이 방향은 폐기 |
| `submissions/baseline_lgbm_seq.csv` | AP 0.05890 / WLL 0.61836 | 미제출 | 기본 `seq` 길이/처음/마지막/고유 토큰 수 피처 추가 |
| `submissions/blend_base80_seq20.csv` | n/a | 0.3405416183 | 베이스라인 80% + seq 20% 블렌드. baseline보다는 개선, 70:30보다는 낮음 |
| `submissions/blend_base70_seq30.csv` | n/a | 0.3405715349 | 베이스라인 70% + seq 30% 블렌드 |
| `submissions/blend_base60_seq40.csv` | n/a | 0.3405770109 | 베이스라인 60% + seq 40% 블렌드. 현재 최고 점수 |
| `submissions/blend_base55_seq45.csv` | n/a | 0.3405770109 | 베이스라인 55% + seq 45% 블렌드. 60:40과 동률 |
| `submissions/blend_base50_seq50.csv` | n/a | 미제출 | 베이스라인 50% + seq 50% 블렌드 후보 |
| `submissions/baseline_lgbm_seq_count.csv` | AP 0.06051 / WLL 0.61784 | 0.339500066 | 기본 seq 피처와 빈도 인코딩 추가. Public에서는 악화 |
| `submissions/blend_best90_seq_count10.csv` | n/a | 미제출 | 현재 최고 90% + seq_count 10% 저위험 블렌드 후보 |
| `submissions/blend_best70_seq_count30.csv` | n/a | 미제출 | 현재 최고 70% + seq_count 30% 블렌드 후보. 단독 count 악화로 위험도 높음 |
| `submissions/blend_best50_seq_count50.csv` | n/a | 미제출 | 현재 최고 50% + seq_count 50% 블렌드 후보. 단독 count 악화로 위험도 높음 |

## 현재 해석

- 이 대회에서는 실제 CTR 사전확률로 보정한 확률보다 클래스 균형 학습의 원본 예측값이 WLL에 더 잘 맞았다.
- 기본 `seq` 피처는 로컬 AP를 조금 올렸지만 WLL은 살짝 나빠졌다. 다만 baseline과 섞으면 공개 리더보드에서 소폭 개선됐다.
- 80:20 < 70:30 < 60:40 순서로 개선됐고, 55:45는 60:40과 동률이다.
- 현재 블렌드 최적점은 seq 비중 40~45% 부근으로 보인다.
- 빈도 인코딩을 추가한 모델은 day7 검증에서 AP 0.06051 / WLL 0.61784로 개선됐지만, Public 점수는 0.339500066으로 크게 악화됐다.
- 현재 count encoding 방식은 day7 validation에는 맞지만 public 분포에는 과적합 또는 분포 불일치가 있는 것으로 본다.
- 다음 우선순위는 기존 최고 블렌드를 유지하면서 count 모델은 낮은 비율 블렌드만 조심스럽게 확인하거나, count encoding 설계를 재검토하는 것이다.
