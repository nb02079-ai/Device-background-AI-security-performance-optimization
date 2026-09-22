# 확인된 인용 목록

**작성일**: 2026-09-21
**원칙**: 본문에 쓰는 모든 수치 인용은 이 목록에서 출처를 확인한 뒤 사용한다. 미확인 인용은 본문에 쓰지 않는다.

---

## 1. CIC-MalMem-2022 선행 연구 보고치 (이진 분류 정확도)

| 출처 | 모델·방법 | 보고 정확도 | 확인 경로 |
|---|---|---|---|
| **Carrier et al. (2022)** — 데이터셋 원 논문 | 스태킹 앙상블 (NB·RF·DT → 로지스틱) | **99%** | ScienceDirect S2667305324001467 에서 인용 확인 |
| Dener et al. (2022) | 빅데이터 환경(PySpark), ML·DL 비교 | 최대 99.9% | Emerald ACI-02-2025-0052 참고문헌 [17] |
| Sensors 23(11):5348 (MDPI, 2023) — "Obfuscated Memory Malware Detection in Resource-Constrained IoT Devices for Smart City Applications" | CompactCBL / RobustCBL | 99.92% / **99.98%** | mdpi.com/1424-8220/23/11/5348 |
| arXiv 2404.02372 (2024) — "Obfuscated Malware Detection: Investigating Real-world…" | RandomForest | **99.99%** | arxiv.org/pdf/2404.02372 |
| arXiv 2602.02184 — "Malware Detection Through Memory Analysis" | XGBoost | 99.98% | arxiv.org/pdf/2602.02184 |
| arXiv 2407.07918 / ScienceDirect (2024) — "Detecting new obfuscated malware variants: A lightweight and interpretable ML approach" | **상위 5개 특징만 사용** | > 99.8%, 파일당 5.7µs | sciencedirect.com/science/article/pii/S2667305324001467 |
| Emerald ACI (2025) — "Obfuscated file-less malware detection using integrating memory forensics data with ML techniques" | 메모리 포렌식 + ML | 최대 99.96% | emerald.com/aci/…/ACI-02-2025-0052 |

### ⚠️ 정정 사항
- 기존 표현 "99.9~**100%**"는 **오류**. 정확도 100% 보고는 없음. 최대치는 **99.99%**
- "100%"로 오기억한 것은 다른 지표: 한 논문의 **AUC-ROC 1.0**, 다른 논문의 **재현율 100%**("detecting all the malware correctly")
- 올바른 서술: **"후속 연구들은 99.9~99.99%의 정확도를 보고했다"**
- **원 논문(Carrier et al.)은 99%** → "선행 연구 전반이 99.9% 이상"이라고 뭉뚱그리면 원 논문까지 포함해 부정확

### 서술 수위
- 선행 연구의 측정이 **틀렸다고 쓰지 않는다** — 그들의 수치는 정확하게 측정된 것
- 본 연구의 주장: "이 높은 수치들은 모델의 우수성보다 **데이터셋의 성질**을 반영할 수 있다"
- 근거: 단일 특징 규칙(`svcscan.nservices`)만으로 99.63%, 깊이 1 트리 99.65%
- arXiv 2407.07918의 "상위 5개 특징으로 99.8%"는 저자들이 경량성의 장점으로 제시 → 본 연구의 관점에서는 "1개로도 충분했다"는 증거로 **재해석 가능**하나, 저자들을 비판하는 형태로 쓰지 않는다

---

## 2. 핵심 방법론 문헌

| 출처 | 용도 | 상태 |
|---|---|---|
| Arp, Quiring, Pendlebury, Warnecke, Pierazzi, Wressnegger, Cavallaro, Rieck. "Dos and Don'ts of Machine Learning in Computer Security." **USENIX Security 2022** | 서론 차별화, 2장 핵심 | ✅ (이전 조사에서 확인) |
| Pendlebury et al. "TESSERACT: Eliminating Experimental Bias in Malware Classification across Space and Time." **USENIX Security 2019** | 시간·공간 편향 | ✅ |
| Shokri et al. "Membership Inference Attacks against Machine Learning Models." **IEEE S&P 2017** | MIA 원전 | ✅ |
| Carlini et al. "Membership Inference Attacks From First Principles." **IEEE S&P 2022** | TPR@low-FPR 평가 원칙 | ✅ |
| Pierazzi et al. "Intriguing Properties of Adversarial ML Attacks in the Problem Space." **IEEE S&P 2020** | 교란 민감도 vs 실제 회피 | ✅ |
| Chicco & Jurman. "The advantages of the MCC over F1 score and accuracy in binary classification evaluation." **BMC Genomics 2020** | 주지표 MCC 근거 | ✅ |
| Barr-Smith et al. "Survivalism: Systematic Analysis of Windows Malware Living-Off-The-Land." **IEEE S&P 2021** | 특징 설계 근거 | ✅ |
| Caruana et al. "Ensemble Selection from Libraries of Models." **ICML 2004** | 스태킹 이득 한계 | ✅ |
| Kuncheva & Whitaker. "Measures of Diversity in Classifier Ensembles." **Machine Learning 2003** | 다양성-성능 무관 | ✅ |
| Frazier. "A Tutorial on Bayesian Optimization." arXiv:1807.02811 (2018) | GP-EI 근거 | ✅ |
| Domingos. "A Unified Bias-Variance Decomposition for Zero-One and Squared Loss." AAAI 2000 | 이진분류 분산 해석 | ✅ |

---

## 2A. 지름길 학습 문헌 (2.2 벤치마크 타당성)

| 출처 | 서지 | 상태 |
|---|---|---|
| **Geirhos et al. (2020)** | Geirhos, R., Jacobsen, J.-H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M., & Wichmann, F. A. Shortcut learning in deep neural networks. *Nature Machine Intelligence*, 2(11), 665–673. doi:10.1038/s42256-020-00257-z | ✅ |
| **Lapuschkin et al. (2019)** | Lapuschkin, S., Wäldchen, S., Binder, A., Montavon, G., Samek, W., & Müller, K.-R. Unmasking Clever Hans predictors and assessing what machines really learn. *Nature Communications*, 10(1), 1096. doi:10.1038/s41467-019-08987-4 | ✅ |

### 핵심 인용 지점
- **Geirhos**: 지름길 = "표준 벤치마크에서는 잘 작동하나 더 까다로운 시험 조건(실제 환경)으로 전이되지 않는 결정 규칙". 저자들은 이것이 **생물·인공을 막론한 학습 시스템 일반의 특성**일 수 있다고 봄
- **Lapuschkin**: "표준 성능 평가 지표는 다양한 문제 해결 방식을 구분하지 못할 수 있다" → **본 연구 중심 주장의 이론적 닻**

### ⚠️ 사용 시 제약
1. **전이 실패를 시험하지 않았다** — Geirhos 정의상 지름길은 전이 실패를 포함. 본 연구는 벤치마크 내 성능만 보였으므로:
   - ❌ "지름길 학습을 입증했다"
   - ✅ "지름길의 전형적 특징을 보인다" / "지름길일 가능성이 높다"
2. **적용 범위** — Geirhos는 심층 신경망 대상. 본 연구의 발견은 결정트리. 저자들 스스로 "학습 시스템 일반의 특성"이라 한 점을 근거로 확장 서술

## 2B. 보안 ML의 벤치마크 타당성 선행 연구 — **직접적 선행 연구**

| 출처 | 서지 | 상태 |
|---|---|---|
| **BiasSeeker** | Wang, C., Xie, X., Wang, T., & Cui, Y. (2026). Bias in the Shadows: Explore Shortcuts in Encrypted Network Traffic Classification. **arXiv:2601.10180** (칭화대, 2026-01-15) | ✅ 확인 / ⚠️ **사전 공개본** — 동료 심사 여부 불명, 인용 시 명시 |
| **Jacobs et al. (2022)** | Jacobs, A. S., et al. AI/ML for Network Security: The Emperor has no Clothes. (2022-11-07) | ⚠️ **학회명 확인 필요** (ACM CCS 2022로 기억하나 미검증) |
| DeGrave et al. | 흉부 X선 COVID-19 탐지기가 거의 완벽한 AUC를 내나 촬영 기관별 표식에 의존 | ⚠️ 미확인 — 비유로 활용 검토 |

### 핵심 인용 지점
- **BiasSeeker**: 모델 무관·데이터 기반 준자동 프레임워크로 암호화 트래픽의 데이터셋 특유 지름길 특징 탐지. 분류기와 무관하게 **"환경에 얽힌(environment-entangled) 특징"** 식별. 공개 데이터셋 19개, NTC 과제 3종으로 평가
  → **CIC-MalMem의 VM 환경 차이 발견과 개념적으로 거의 동일**
- **Jacobs et al.**: 체계화를 통해 오래된 데이터셋 의존·설계상 간과·근거 없는 가정의 결과를 보임. 제안된 암호화 트래픽 분류기 대다수가 레거시 데이터셋 때문에 실제로는 **비암호화 트래픽**을 사용

### ⚠️ 2.2 포지셔닝 수정 (중요)
- ❌ 폐기: "특정 공개 벤치마크의 편향을 정량화한 연구는 드묾"
- ✅ 수정: "벤치마크 타당성 문제는 네트워크 트래픽 분류에서 확인되어 왔다[Jacobs; Wang]. 본 연구는 이 문제의식을 **호스트 행위·메모리 기반 악성코드 탐지 벤치마크**로 확장하고, 이를 모델·개선·해석 층위와 연결한다."
- 사유: "처음이다"는 반례 하나에 무너지나, "기존 흐름을 새 영역으로 확장"은 방어 가능

### 본 연구의 차별점

| | BiasSeeker / Jacobs et al. | 본 연구 |
|---|---|---|
| 영역 | 네트워크 트래픽 | **호스트 메모리·행위** |
| 범위 | 벤치마크 타당성 자체 | 벤치마크를 **네 층위 중 하나**로, 나머지와 연결 |
| 측정 | 지름길 특징을 **탐지** | 교란이 성능에 **기여하는 크기**를 통제 실험으로 분리 (약 0.03 MCC) |

### 교훈 (기록)
"드묾"이라는 주장을 검색 한 번으로 반박할 선행 연구가 존재했다. **"없다/드물다"류 주장은 반드시 검색으로 확인 후 사용**하며, 확인 범위를 한정어로 명시한다.

## 3. 데이터셋

| 데이터셋 | 인용 | 상태 |
|---|---|---|
| **DMBD 2025** | Lawrence Livermore National Laboratory. "Dynamic Malware Behaviorial Dataset." July 2025. LLNL-CODE-837816. (Wintap 기반, gdo-wintap.llnl.gov) | ✅ README 원문 확인 |
| **CIC-MalMem-2022** | Carrier et al. (2022). 원 논문 제목·학회 **최종 확인 필요** (ICISSP 2022, "Detecting Obfuscated Malware using Memory Feature Engineering"로 기억하나 미검증) | ⚠️ 제목·학회 재확인 |

---

## 4. 미확인 — 본문 사용 전 확인 필요

| 항목 | 사유 |
|---|---|
| Carrier et al. (2022) 정확한 서지 | 제목·학회명 원문 미확인 |
| Dener et al. (2022) 정확한 서지 | 2차 인용으로만 확인 |
| 업계 수치 (CrowdStrike 1~3% CPU 등, §7.1) | 벤더 자료이므로 인용 방식 결정 필요 |
