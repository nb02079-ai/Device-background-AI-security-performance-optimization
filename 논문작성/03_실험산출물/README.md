# Phase 3~6 산출 파일 패키지

**생성일**: 2026-09-20
**대상 프로토콜**: v3.5
**데이터셋**: DMBD 2025 (LLNL) — 집계본 `dmbd_event_counts.csv` (64,747 실행)

---

## 디렉터리 구조

```
phase3/
├── scripts/          실행 스크립트 5종
│   ├── prep.py           데이터 준비·분할·중복제거 (§2.1, §3.3.2, §3.4)
│   ├── gpbo.py           GP-EI 베이지안 최적화 클래스 (§4.2)
│   ├── run_bo.py         BO 탐색공간·목적함수·재개 가능 실행기
│   ├── run_models.py     모델 A/B/C/D 학습 (재개 가능)
│   └── summarize.py      성능표·차이 CI·임계값 분석
├── results/
│   ├── featureset.json   전체/제거/최종 특징, 중복쌍 ρ, 봉인 해시, 분할 규모
│   ├── bo_result.json    BO 최적 파라미터·30회 이력·알고리즘 표기
│   ├── bo_state.pkl      BO 중간 상태 (X, y, RNG) — 재현용
│   └── results_all.json  모델×시드별 전체 지표 (MCC/F1/P/R/PR-AUC/FPR/혼동행렬)
├── indices/          분할 인덱스 (원본 CSV의 행 번호)
│   ├── tr_idx.npy        Train 24,000
│   ├── va_idx.npy        Validation 8,000
│   ├── h5_idx.npy        H5-Test 8,000        ⚠️ 봉인
│   └── offtest_idx.npy   공식 Test 24,747     ⚠️ 봉인
├── probs/            Validation 예측 확률 (모델 4종 × 시드 5개 = 20개)
└── old_run/          구 분할(Train 30,000) 결과 보존

phase6/
├── scripts/
│   ├── step1_lightweight.py   경량화 선택 (두 경로 공유)
│   ├── step23.py              정규화·이산화 선택 + MIA 분할 무결성 검증
│   ├── step4_perturb.py       교란학습 증강비율 선택
│   ├── h5_eval_part1.py       H5-Test 봉인해제·탐지/지연/강건성
│   ├── h5_eval_part2_mia.py   고정 감사셋 MIA
│   └── official_test.py       공식 Test 코호트별 보고 + 2024 AUC 진단
└── results/                   각 스크립트 결과 JSON

phase4/
├── scripts/
│   ├── bench_resource.py 리소스 벤치마크 (§7)
│   ├── mia.py            멤버십 추론 공격 (§5)
│   └── perturb.py        교란 민감도 + 순열 중요도 (§6, §6.4)
└── results/
    ├── resource.json     배치별 지연·처리량·메모리
    ├── env.json          측정 환경 (CPU 수, 메모리, 스레드)
    ├── mia.json          selection/audit 분리 결과 + 확신도 격차
    ├── perturb.json      특징별 단계별 flip rate·확률감소폭, 순열 중요도, H2 상관
    ├── year_confound.json 연도 예측 R²·AUC, 특징별 연도 상관
    ├── cohort_check.json  코호트별 MCC·AUC·연도단독 AUC
    ├── mia_audit_mem.npy MIA 최종 감사셋 member ID   ⚠️ 고정
    └── mia_audit_non.npy MIA 최종 감사셋 non-member ID ⚠️ 고정
```

---

## 봉인 셋 — Phase 6에서 해제 완료 (2026-09-21)

아래 두 셋은 Phase 6에서 **해시 검증 후 1회 해제**되었다. 두 해시 모두 일치했다.

| 파일 | 내용 | 무결성 해시 |
|---|---|---|
| `phase3/indices/h5_idx.npy` | H5-Test 8,000 | `47254fd15bb1f192` |
| `phase3/indices/offtest_idx.npy` | 공식 Test 24,747 | `c63d8ca03e7a4480` |

해시는 인덱스를 정렬 후 SHA-256의 앞 16자다. H5-Test는 H5 판정에, 공식 Test는 시간적 일반화 보고에만 사용되었다.

`phase4/results/mia_audit_*.npy`는 MIA 최종 감사셋으로, Phase 4에서 1회 사용 후 고정되었다(§5.1).

---

## 재현 절차

원본 CSV(`dmbd_event_counts.csv`)를 `/mnt/user-data/uploads/`에 두고 아래 순서로 실행한다.

```bash
# Phase 3
python3 prep.py                       # 분할·중복제거·봉인 해시 기록
python3 run_bo.py 14                  # BO 30회 (재개 가능, 여러 번 호출)
python3 run_bo.py 14
python3 run_bo.py 5
python3 run_models.py A_RF            # 모델별 시드 5회
python3 run_models.py B_RF_BO
python3 run_models.py D_MLP
python3 run_models.py C_STACK 42      # 스태킹은 시드별 분할 실행 권장
python3 summarize.py

# Phase 4
python3 bench_resource.py
python3 mia.py A_RF                   # 모델별 실행
python3 perturb.py A_RF
```

### 재현성 보장 장치

- **고정 시드**: 분할 42, 모델 학습 {42, 7, 123, 2024, 777}, BO 42, OOF CV 42
- **BO 재개 시 RNG 복원**: `bo_state.pkl`에 난수 상태까지 저장하므로 중단·재개가 결과에 영향을 주지 않는다
- **결정론적 중복 제거**: Spearman |ρ|>0.95 쌍에서 컬럼 순서상 뒤를 제거

### 환경

| 항목 | 값 |
|---|---|
| Python | 3.12 |
| scikit-learn | 1.8.0 |
| numpy / pandas / scipy | 2.4.4 / 3.0.2 / 1.17.1 |
| psutil | 7.2.2 |
| CPU / RAM | 1 코어 / 3.9GB (`n_jobs=1` 고정) |

`hyperopt`·`optuna`·`xgboost`·`lightgbm`은 미설치 상태이며, 이 때문에 BO를 GP-EI로 직접 구현하고 스태킹 베이스를 `HistGradientBoostingClassifier`로 대체했다(프로토콜 §4.2 변경이력 참고).

---

## 주요 결과 요약

### Phase 3 — Validation 성능 (시드 5회 평균)

| 모델 | MCC | FPR | 지연(batch=10) |
|---|---|---|---|
| A: RF 기본 | 0.8246 | 0.0814 | 1.2209ms |
| **B: RF+BO** | **0.8342** | **0.0655** | 0.2184ms |
| C: 스태킹 | 0.8325 | 0.0689 | 0.7243ms |
| D: MLP | 0.6267 | 0.0906 | **0.0045ms** |

BO 최적값: `n_estimators=54, max_depth=50, max_features=2, min_samples_split=11, min_samples_leaf=1, criterion=gini` (Train 내부 3-fold CV MCC 0.8285)

### Phase 4 — 3대 진단

| 모델 | MIA AUC_adv [95% CI] | 최대 flip rate(30%) | H2 Spearman ρ |
|---|---|---|---|
| A | **0.5720** [0.5544, 0.5897] | 0.0645 | +0.750 (p=0.052) |
| B | 0.5259 [0.5080, 0.5437] | 0.0620 | +0.536 (p=0.215) |
| C | 0.5230 [0.5052, 0.5409] | 0.0536 | +0.536 (p=0.215) |
| D | 0.5001 [0.4822, 0.5180] | 0.0156 | +0.893 (p=0.007) |

### 간접 교란 검증 (§11.2.4)

행위 특징 12개로 연도 예측 시 R²=0.6386 / "2017?" AUC=0.9430. 라벨 고정 시에도 악성 내부 R²=0.5782.
다만 연도단독 AUC를 0.5000으로 만든 조건에서도 **MCC 0.7788 / AUC 0.9500 유지** → 교란 기여 약 0.03 MCC.

---

## 관련 문서

| 문서 | 내용 |
|---|---|
| `protocol_v3.0.md` | 실험 프로토콜 (변경이력 55건 포함) |
| `phase3_model_implementation.md` | Phase 3 전체 구현 (모델 A·D 중심) |
| `phase3_models_BC_implementation.md` | 모델 B·C 상세 (BO 원리, OOF 스태킹) |
| `phase4_diagnostics.md` | Phase 4 진단 실험 상세 |
| `interim_analysis_phase0-3.md` | 중간 분석 (2026-09-19) |
| `phase5_report_H4_abandonment.md` | Phase 5 — H4 폐기 및 간접 교란 검증 |
| `phase6_report.md` | Phase 6 — H5 기각, 교란학습 결함, 시간적 일반화, B 해석 정정 |

---

## 포함하지 않은 파일

- `df.pkl` (14MB) — 원본 CSV의 pickle 캐시. `prep.py` 실행 시 자동 생성되므로 제외
- `models_ad.py`, `models_bcd.py` — `run_models.py`로 통합되어 폐기된 초기 스크립트

### Phase 6 — H5 판정: **기각**

| 모델 (H5-Test) | MCC | MIA AUC_adv | 최대 flip | 지연 |
|---|---|---|---|---|
| 선택적 | **0.8280** | 0.5599 | 0.0544 | 0.1347ms |
| 일괄 | 0.7868 | **0.5012** | **0.0052** | 0.1336ms |

목표축(지연) 비열등·MCC 우위이나 **MIA 축 열등**으로 지지 조건 불충족. 두 전략은 트레이드오프 관계.
