# FabPulse — 실제 UCI SECOM 검증 결과 (2026-11-17)

## 데이터 출처 및 무결성
- UCI SECOM, McCann & Johnston (2008), DOI: 10.24432/C54305, CC BY 4.0.
- 제조 샘플 1,567건, 익명 수치형 변수 590개, 불량 104건 (6.64%).
- 확인된 원본 SHA256:
  - secom.data: `20f0e7ee434f7dcbae0eea9ffff009a2b57f42d6b0dc9a5bd4f00782c0a3374c`
  - secom_labels.data: `126884cf453705c9e61a903fe906f0665a3b45ce3639e621edc5c93c89627e03`
  - secom.names: `6d91b0b46cdee03064ee3e3112f937c1b3f7fcd9933575794ec07974e6f1ea59`

## 절차
- 검사 시각 기준 정렬, 시간순 60% 학습 / 20% 검증 / 20% 홀드아웃 평가.
- 결측값 중앙값 대체 → 상수 변수 제거 → SelectKBest(40개 변수).
- Logistic Regression과 Random Forest 비교.
- 검증 구간 Recall@Top10%, 동률 시 Average Precision으로 모델 선택. 최종 평가 구간은 모델 선택에 사용하지 않음.

## 검증 구간 (313건, 실제 불량 11건)
| 모델 | 상위 10% 점검 중 포착 불량 | Recall@10% | AP | ROC-AUC |
|---|---:|---:|---:|---:|
| Random Forest | 2/11 | 18.18% | 0.0719 | 0.6240 |
| Logistic Regression | 2/11 | 18.18% | 0.0527 | 0.5361 |

## 최종 홀드아웃 테스트 (314건, 실제 불량 17건)
- 선택 모델: Random Forest
- 상위 위험도 32건 점검: 실제 불량 3건 포착 (3/32 = 9.38% 점검 정밀도).
- **Recall@Top10%: 17.65%** (3/17).
- **Lift@Top10%: 1.7316배** (무작위 점검 대비 이론적 기대값과의 비교).
- **Average Precision (PR-AUC): 0.09618**, **ROC-AUC: 0.68509**.
- 해당 기간 불량 비율: **5.41%**.
- 실제 불량 17건 중 14건은 상위 10% 선별에서 놓침. 실무 배포·예방 효과 검증 결과로 해석 불가.

## 해석과 한계
- 제한된 점검률에서 무작위 선별보다 높은 불량 집중도가 관찰됨.
- 홀드아웃 불량이 17건에 불과해 통계적 불확실성이 큼. 시간 순 분할, 데이터 분포 차이에도 민감할 수 있음.
- 익명 측정 변수만 존재하여 공정명, 장비 ID, 결함 원인을 확인할 수 없음.
- 검사 시각 기준 분할이며 실제 제조 시점·선행 조건이 검증되지 않아 사전 예방/실시간 경보를 입증하지 않음.
- 관측 데이터 기반 분류 프로토타입이며 SK하이닉스의 실제 FAB 운영 성과와 무관함.

## 재현
- 워크플로 검증 성공: https://github.com/rlaqhfla0917-bit/rlaqhfla0917-bit/actions/runs/44012267006
- 해당 실행의 `secom-validation` 아티팩트에 `metrics.json`, `test_predictions.csv`, `feature_importance.csv` 저장.
- 로컬 실행: `python scripts/download_data.py && python train.py`.
