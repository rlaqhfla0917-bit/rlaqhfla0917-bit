# FabPulse — 반도체 제조공정 불량 위험 분류 및 점검 우선순위 지원

> **상태:** 구현된 분석 코드 + Streamlit UI / 공식 SECOM 데이터 다운로드·실제 학습은 실행 환경에 따라 별도 수행  
> **용도:** SK하이닉스 Fab Operation Engineering 관련 기술 포트폴리오의 재현 가능한 연구용 프로토타입  
> **중요:** SK하이닉스와 무관한 공개 데이터셋을 사용하며, 현장 운영에 연결된 상용 시스템이 아닙니다.

## 문제정의

반도체 제조 과정의 590개 익명화 측정 변수를 활용하여, 전체 샘플 중 **불량 위험 점수가 높은 상위 10%를 먼저 점검할 때 실제 불량을 얼마나 포착할 수 있는가?**

- 단순 분류 정확도 대신 `Recall@Top10%`, `Lift@Top10%`, `Average Precision (AP)`로 평가합니다.
- `Lift@Top10% = Precision@Top10% / 실제 불량 비율`입니다. 1.0배면 무작위로 선택하는 것과 같습니다.
- 품질 점검률 10%는 사용자 선택이 가능한 **가상의 점검 자원 제약**입니다. 실제 FAB 검사정책이 아닙니다.
- 예측 모델은 **보정된 불량 확률이 아닌 상대 위험 점수**로 샘플을 순위화합니다.

## 사용 데이터 및 확인 가능한 사실

**UCI SECOM**, McCann & Johnston (2008)  
공식 페이지: https://archive.ics.uci.edu/dataset/179/secom  
DOI: https://doi.org/10.24432/C54305  
라이선스: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — 원 저작자 표시 필수

- 제조 샘플 1,567개 / 수치형 입력 변수 590개 / 불량 104개
- `secom.data`: 590개 측정값, 결측값 포함
- `secom_labels.data`: `-1`=양품, `1`=불량 및 검사 시각
- **존재하지 않는 정보:** 실제 센서 이름·단위, 공정 단계, 장비 ID, 공정 작업 시각, 실제 결함 원인, 점검·재작업 비용, 실시간 스트림
- 따라서 **원인 확정 / 특정 설비 고장 예지 / 실시간 공정제어 / 사전 예측 효용 / 수율 개선 실적**을 주장하지 않습니다.

## 실행 방법 (Windows PowerShell)

```powershell
cd FabPulse_SECOM
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/download_data.py
python train.py
streamlit run app.py
```

macOS/Linux:

```bash
cd FabPulse_SECOM
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py
python train.py
streamlit run app.py
```

인터넷 없는 환경에서도 `streamlit run app.py`로 **합성 데이터 기반 데모 화면**을 볼 수 있습니다. 화면 상단에 **합성 시연 데이터임을 명시**합니다. 시연 수치나 데모 중요도를 분석 성과로 쓰지 마세요.

다운로드 스크립트가 실패하면 아래 공식 링크에서 직접 세 파일을 받아 `data/raw/` 폴더에 넣을 수 있습니다.
- https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data
- https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom_labels.data
- https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.names

## 재현 가능한 분석 방법

1. `secom.data`와 레이블의 행 순서 일치를 검증하고, 별도로 제공된 검사 시각을 오름차순 정렬합니다.
2. **시간순 60% 학습 / 20% 검증 / 20% 최종 평가**로 분리합니다. 데이터가 빈약한 구간에서 평가는 불안정할 수 있습니다.
3. `SimpleImputer(median) -> VarianceThreshold -> SelectKBest(f_classif, k=40)`를 학습 파이프라인에 포함하여 누수를 방지합니다.
4. 클래스 가중치가 적용된 **Logistic Regression**과 **Random Forest**를 비교합니다.
5. 검증 구간의 `Recall@Top10%`를 최우선으로, 동률이면 AP로 모델을 선택합니다.
6. 학습+검증 구간에서 재학습 후 별도 테스트에서 **단 한 번** 성능을 측정합니다.
7. `artifacts/metrics.json`, `test_predictions.csv`, `feature_importance.csv`, `model.joblib`를 저장합니다.
8. Streamlit에서 상위 점검 비율을 조절하여 불량 포착률과 점검 효율 변화를 관찰합니다.

> 주의: 이 프로젝트의 데이터는 관측 기반이며, 시간순으로 분리하더라도 생산 중 측정 선후관계가 보장되지 않습니다. 과거 품질 판정에 대한 **소급 분류(retrospective screening)**로 해석하십시오. 분석에 사용한 모델의 점수는 교정된 불량 확률이 아닙니다.

## 구조

```text
FabPulse_SECOM/
├── app.py                     # Streamlit 대시보드(미학습 시 합성 데모)
├── train.py                   # 학습·검증·보류 테스트 파이프라인
├── src/
│   ├── data.py                # 파일 파싱·검사 시각 정렬
│   └── modeling.py            # 분류 모델·점검 효율 지표·변수 중요도
├── scripts/
│   ├── download_data.py       # 공식 데이터 다운로드 및 SHA256 확인
│   └── self_check.py          # 합성 데이터 기반 코드 통합 테스트
├── data/raw/                  # 원본 다운로드 위치 (Git 추적 제외)
├── artifacts/                 # 학습 후 자동 생성 (Git 추적 제외)
├── .streamlit/config.toml
├── requirements.txt
└── README.md
```

## 실제 결과를 이력서/자기소개서에 활용할 때

먼저 `python train.py`를 **공식 SECOM 자료로 실행한 다음** 출력되는 보류 테스트의 AP·Recall@10%·Lift@10% 수치를 확인하세요. 코드 작성이나 시연용 데이터 평가만으로 '불량률 감소', '수율 개선', 'AI 성능 달성' 등을 주장하면 안 됩니다.

권장 문장 틀: 'UCI 공개 반도체 제조 데이터 1,567건의 익명 측정 변수 590개를 이용하여 불균형 데이터 전처리와 분류 모델을 구현하고, 시간순 분할과 불량 포착률·점검 효율로 평가했습니다.'

## 데이터 라이선스 및 인용

McCann, M. & Johnston, A. (2008). *SECOM* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C54305 (CC BY 4.0). 이 프로젝트의 별도 코드와 UI는 교육용 예시입니다.
