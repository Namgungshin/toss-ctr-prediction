# 실험 기록

| 파일 | 로컬 검증 | 공개 리더보드 점수 | 메모 |
|---|---:|---:|---|
| `submissions/baseline_lgbm.csv` | AP 0.05833 / WLL 0.61761 | 0.340445203 | `seq`를 제외한 첫 LightGBM 원본 예측 베이스라인 |
| `submissions/baseline_lgbm_prior_calibrated.csv` | n/a | 0.2045772218 | 실제 CTR 사전확률 보정은 점수를 크게 악화. 이 방향은 폐기 |
| `submissions/baseline_lgbm_seq.csv` | AP 0.05890 / WLL 0.61836 | 미제출 | 기본 `seq` 길이/처음/마지막/고유 토큰 수 피처 추가 |
| `submissions/blend_base80_seq20.csv` | n/a | 0.3405416183 | 베이스라인 80% + seq 20% 블렌드. baseline보다는 개선, 70:30보다는 낮음 |
| `submissions/blend_base70_seq30.csv` | n/a | 0.3405715349 | 베이스라인 70% + seq 30% 블렌드. 현재 최고 점수 |
| `submissions/blend_base50_seq50.csv` | n/a | 미제출 | 베이스라인 50% + seq 50% 블렌드 후보 |

## 현재 해석

- 이 대회에서는 실제 CTR 사전확률로 보정한 확률보다 클래스 균형 학습의 원본 예측값이 WLL에 더 잘 맞았다.
- 기본 `seq` 피처는 로컬 AP를 조금 올렸지만 WLL은 살짝 나빠졌다. 다만 baseline과 섞으면 공개 리더보드에서 소폭 개선됐다.
- 80:20보다 70:30이 더 좋아서 현재 블렌드 최적점은 seq 비중 20~40% 구간에 있을 가능성이 있다.
- 다음 우선순위는 60:40 블렌드 확인, 안정적인 범주형 그룹에 대한 빈도 인코딩, 타깃 인코딩이다.
