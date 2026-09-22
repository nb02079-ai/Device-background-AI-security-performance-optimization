# Phase 3 모델 구현 문서

**작성일**: 2026-09-19 (2026-09-20 분할 변경 반영 재작성)
**대상 프로토콜**: v2.9
**데이터**: DMBD 2025 (`dmbd_event_counts.csv`, 64,747 실행)
**목적**: 모델 4종(A/B/C/D)의 구현 전문과 실행 결과를 재현 가능한 형태로 기록한다.

> **봉인 원칙**: 본 Phase의 모든 평가는 **Validation(8,000)**에서 수행되었다. **H5-Test(8,000)**와 **공식 Test(24,747)**는 인덱스를 저장하고 무결성 해시만 기록했을 뿐, 어떤 코드에서도 로드하지 않는다(프로토콜 §9).
>
> 봉인 해시 — H5-Test `47254fd15bb1f192`, 공식 Test `c63d8ca03e7a4480`
>
> **모델 B·C의 상세 구현**(베이지안 최적화 원리, OOF 스태킹)은 별도 문서 `phase3_models_BC_implementation.md` 참고.

---

## 0. 실행 환경

| 항목 | 값 |
|---|---|
| Python | 3.12 |
| scikit-learn | 1.8.0 |
| numpy / pandas / scipy | 2.4.4 / 3.0.2 / 1.17.1 |
| 미설치 (네트워크 차단) | `hyperopt`, `skopt`, `optuna`, `xgboost`, `lightgbm` |

`hyperopt` 부재로 프로토콜 §4.2의 BO-TPE를 사용할 수 없어 **GP 기반 BO를 직접 구현**했다(§2 참고).

---

## 1. 데이터 준비 — `prep.py`

프로토콜 §3.4의 처리 순서를 그대로 구현한다.

```python
"""Phase 3 - 데이터 준비 (프로토콜 §3.4 처리 순서 준수)"""
import pandas as pd, numpy as np, json, hashlib
from sklearn.model_selection import train_test_split

SEED_SPLIT = 42
ROOT = '/mnt/user-data/uploads/'
OUT = '/home/claude/phase3/'
import os; os.makedirs(OUT, exist_ok=True)

# --- 1. 로드 ---
df = pd.read_csv(ROOT + 'dmbd_event_counts.csv')

# --- 2. 제외 특징 제거 (§3.3.2) ---
EXCLUDE = ['detonation', 'dur_after_detonation_s']
META = ['experiment', 't_first', 't_last', 't_detonate', 'year', 'label', 'test']
FEATURES = [c for c in df.columns if c not in EXCLUDE + META]

y = (df['label'] == 'malicious').astype(int)

# --- 3. 공식 분할 보존 + 공식 train을 3분할 (§2.1) ---
is_test = df['test'].astype(bool)
dev_idx = df.index[~is_test]        # 공식 train 40,000
off_test_idx = df.index[is_test]    # 공식 test 24,747 (봉인)

tr_idx, rest_idx = train_test_split(
    dev_idx, train_size=0.60, random_state=SEED_SPLIT, stratify=y.loc[dev_idx])
va_idx, h5_idx = train_test_split(
    rest_idx, train_size=0.50, random_state=SEED_SPLIT, stratify=y.loc[rest_idx])

# 봉인 셋 무결성 해시 (§9)
def idx_hash(idx):
    return hashlib.sha256(np.sort(np.asarray(idx)).tobytes()).hexdigest()[:16]
seal = {'H5_Test': idx_hash(h5_idx), 'official_Test': idx_hash(off_test_idx)}

# --- 4. Train에서만 중복 특징 판단·제거 (§3.3.2) ---
Xtr_raw = df.loc[tr_idx, FEATURES]
corr = Xtr_raw.corr(method='spearman').abs()
drop_dup, pairs = [], []
for i, a in enumerate(FEATURES):
    if a in drop_dup: continue
    for b in FEATURES[i+1:]:
        if b in drop_dup: continue
        r = corr.loc[a, b]
        if r > 0.95:
            pairs.append((a, b, round(float(r), 4)))
            drop_dup.append(b)   # 컬럼 순서상 뒤의 특징 제거 (결정론적)

FEAT_FINAL = [c for c in FEATURES if c not in drop_dup]

np.save(OUT+'tr_idx.npy', tr_idx.values)
np.save(OUT+'va_idx.npy', va_idx.values)
np.save(OUT+'test_idx.npy', test_idx.values)
json.dump({'features_all': FEATURES, 'dropped_dup': drop_dup,
           'dup_pairs': pairs, 'features_final': FEAT_FINAL},
          open(OUT+'featureset.json','w'), ensure_ascii=False, indent=2)
df.to_pickle(OUT+'df.pkl')
```

### 실행 결과

```
Train       24000  악성 50.0%
Validation   8000  악성 50.0%
H5-Test      8000  악성 50.0%   [봉인]
공식 Test   24747  악성 78.8%   [봉인]

봉인 해시: {'H5_Test': '47254fd15bb1f192', 'official_Test': 'c63d8ca03e7a4480'}

중복 쌍: [('n_events','n_uniq_child',0.9829), ('n_events','n_uniq_cguid',1.0)]
제거: ['n_uniq_child', 'n_uniq_cguid']
최종 특징 12개
```

**최종 특징 12개**: `file_create`, `image_load`, `net_connect`, `process_create`, `process_term`, `reg_create_key`, `reg_delete_key`, `reg_delete_value`, `reg_write`, `n_events`, `duration_s`, `n_uniq_parent`

> **누수 방지 확인 지점**: 중복 판단은 `df.loc[tr_idx, FEATURES]`로 **Train만** 사용한다. `test_idx`는 저장만 하고 로드하지 않는다.

---

## 2. GP 기반 베이지안 최적화 — `gpbo.py`

프로토콜 §4.2의 TPE를 대체하는 구현. Matérn(ν=2.5) 커널 GP에 Expected Improvement 획득함수를 사용한다.

```python
"""GP 기반 베이지안 최적화 (Expected Improvement)"""
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel


class GPBayesOpt:
    """연속/정수/범주 혼합 탐색공간에 대한 GP-EI 베이지안 최적화.

    space: dict[name] = ('int', lo, hi) | ('cat', [values])
    모든 차원을 [0,1]로 정규화해 GP를 적합하고, EI를 최대화하는 점을 다음 시행으로 선택.
    """

    def __init__(self, space, n_init=10, n_iter=20, seed=42):
        self.space = space
        self.names = list(space.keys())
        self.n_init = n_init
        self.n_iter = n_iter
        self.rng = np.random.RandomState(seed)
        self.X, self.y = [], []

    # --- 정규화 좌표 <-> 실제 하이퍼파라미터 ---
    def _decode(self, u):
        p = {}
        for k, ui in zip(self.names, u):
            spec = self.space[k]
            if spec[0] == 'int':
                lo, hi = spec[1], spec[2]
                p[k] = int(np.clip(np.floor(lo + ui * (hi - lo + 1)), lo, hi))
            else:  # cat
                vals = spec[1]
                p[k] = vals[int(np.clip(np.floor(ui * len(vals)), 0, len(vals) - 1))]
        return p

    def _sample_u(self, n):
        return self.rng.rand(n, len(self.names))

    # --- Expected Improvement ---
    def _ei(self, gp, U, best, xi=0.01):
        mu, sd = gp.predict(U, return_std=True)
        sd = np.maximum(sd, 1e-9)
        imp = mu - best - xi
        z = imp / sd
        return imp * norm.cdf(z) + sd * norm.pdf(z)
```

**EI 수식**: 관측 최고값 `f*`에 대해

```
EI(x) = (μ(x) − f* − ξ)·Φ(z) + σ(x)·φ(z),   z = (μ(x) − f* − ξ)/σ(x)
```

`ξ = 0.01`은 탐색(exploration) 가중치다. 첫 항은 개선 기대값, 둘째 항은 불확실성 보너스로, 이 둘의 균형이 BO의 핵심이다.

### 탐색 공간 및 목적함수 — `run_bo.py` 발췌

```python
SPACE = {
    'n_estimators':     ('int', 10, 200),
    'max_depth':        ('int', 5, 50),
    'max_features':     ('int', 1, len(FEAT)),   # §4.2: 상한을 실제 특징 수(12)로 조정
    'min_samples_split':('int', 2, 11),
    'min_samples_leaf': ('int', 1, 11),
    'criterion':        ('cat', ['gini', 'entropy']),
}
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

def objective(p):
    """Train 내부 3-fold CV의 MCC (Test 미사용, §8 주지표와 일치)"""
    scores = []
    for itr, iva in cv.split(Xtr, ytr):
        m = RandomForestClassifier(**p, random_state=42, n_jobs=-1)
        m.fit(Xtr[itr], ytr[itr])
        scores.append(matthews_corrcoef(ytr[iva], m.predict(Xtr[iva])))
    return float(np.mean(scores))
```

> **재개 가능 설계**: 실행 환경의 시간 제약으로 매 평가 후 상태(`X`, `y`, RNG)를 `bo_state.pkl`에 저장하고, 재호출 시 이어서 진행하도록 구현했다. 난수 상태까지 보존하므로 중단 여부가 결과에 영향을 주지 않는다.

### BO 탐색 결과 (30회)

| 단계 | 최고 CV MCC | 비고 |
|---|---|---|
| 초기 무작위 10회 | 0.8288 | |
| BO 20회 | **0.8360** | 반복 3회차에 갱신 |

**최적 하이퍼파라미터**

```python
{'n_estimators': 88, 'max_depth': 25, 'max_features': 4,
 'min_samples_split': 2, 'min_samples_leaf': 2, 'criterion': 'entropy'}
```

관측된 CV MCC 범위는 0.7043~0.8360이었다. 최악값은 `max_depth=6`, `max_features=1`처럼 표현력이 과도하게 제한된 조합에서 나왔다.

---

## 3. 모델 4종 학습 — `run_models.py`

```python
def evaluate(yt, yp, yprob):
    tn, fp, fn, tp = confusion_matrix(yt, yp).ravel()
    return dict(MCC=matthews_corrcoef(yt, yp), F1=f1_score(yt, yp),
                Precision=precision_score(yt, yp), Recall=recall_score(yt, yp),
                PR_AUC=average_precision_score(yt, yprob), FPR=fp/(fp+tn),
                TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp))
```

### 모델 A — RandomForest 기본

```python
m = RandomForestClassifier(n_estimators=300, random_state=s, n_jobs=-1)
m.fit(Xtr, ytr)
prob = m.predict_proba(Xva)[:, 1]
```

`n_estimators=300`은 **본 연구 고정 기준값**이다(라이브러리 기본값은 100). 그 외는 sklearn 기본값. 스케일링을 적용하지 않는다 — 트리는 임계값 비교만 하므로 단위가 무관하다.

### 모델 B — RandomForest + GP-BO

```python
BP = json.load(open(OUT + 'bo_result.json'))['best_params']
m = RandomForestClassifier(**BP, random_state=s, n_jobs=-1)
m.fit(Xtr, ytr)
prob = m.predict_proba(Xva)[:, 1]
```

### 모델 C — 스태킹 (OOF 5-fold)

```python
bases = [RandomForestClassifier(**BP, random_state=s + k, n_jobs=-1) for k in range(3)]
bases.append(HistGradientBoostingClassifier(random_state=s))
oof_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# OOF 예측으로 메타 학습 (§4.2.1 in-sample 예측 금지)
oof = np.column_stack([
    cross_val_predict(b, Xtr, ytr, cv=oof_cv, method='predict_proba', n_jobs=1)[:, 1]
    for b in bases])
meta = LogisticRegression(max_iter=1000).fit(oof, ytr)

for b in bases:
    b.fit(Xtr, ytr)
prob = meta.predict_proba(
    np.column_stack([b.predict_proba(Xva)[:, 1] for b in bases]))[:, 1]
```

> **핵심 구현 지점**: `cross_val_predict`로 **자기가 학습하지 않은 fold에 대한 예측(OOF)**만 메타모델 학습에 쓴다. 선행 구현(IDS-ML)이 `base.predict(X_train)`으로 in-sample 예측을 사용해 데이터 누수를 일으킨 지점이며, 프로토콜 §4.2.1의 의도적 상이점에 해당한다.

### 모델 D — MLP

```python
sc = StandardScaler().fit(Xtr)          # Train에서만 fit (§3.4)
m = MLPClassifier(max_iter=200, early_stopping=True,
                  n_iter_no_change=10, random_state=s)   # sklearn 기본 (100,)
m.fit(sc.transform(Xtr), ytr)
prob = m.predict_proba(sc.transform(Xva))[:, 1]
```

**튜닝하지 않는다.** 모델 A도 미튜닝이므로 A↔D 구조 비교의 대칭성을 확보하기 위함이며, 임의값 대신 라이브러리 기본값을 근거로 삼아 선택의 자의성을 제거했다.

> MLP만 스케일링이 필요하다. 트리와 달리 가중치 곱셈 구조라 `n_events`(수천 단위)와 `reg_delete_value`(한 자리)의 스케일 차이가 학습을 지배하기 때문이다.

---

## 4. 실행 결과

### 4.1 Validation 성능 (시드 5회 평균±표준편차, 임계값 0.5)

| 모델 | MCC | F1 | Precision | Recall | PR-AUC | FPR | 평균 혼동행렬 [[TN,FP],[FN,TP]] |
|---|---|---|---|---|---|---|---|
| A: RF 기본 | 0.8246±0.0008 | 0.9117 | 0.9175 | 0.9060 | 0.9700 | 0.0814 | [[3674,326],[376,3624]] |
| **B: RF+BO** | **0.8342±0.0015** | 0.9154 | **0.9321** | 0.8992 | 0.9725 | **0.0655** | [[3738,262],[403,3597]] |
| C: RF+BO+스태킹 | 0.8325±0.0009 | 0.9148 | 0.9290 | 0.9011 | **0.9740** | 0.0689 | [[3724,276],[396,3604]] |
| D: MLP | 0.6267±0.0065 | 0.7845 | 0.8861 | 0.7039 | 0.8863 | 0.0906 | [[3637,363],[1184,2816]] |

학습 시간(시드당): A 6.3초 / B **0.9초** / C 15.0초 / D 3.0초

### 4.2 차이의 95% 신뢰구간 (대응 비교, §8)

| 비교 | Δ MCC | 95% CI | 판정 |
|---|---|---|---|
| 튜닝 효과 (A→B) | +0.0097 | [+0.0069, +0.0124] | 차이 관측됨 |
| 스태킹 효과 (B→C) | −0.0017 | **[−0.0045, +0.0011]** | **유의미한 차이 발견 못함** |
| 튜닝+스태킹 (A→C) | +0.0080 | [+0.0061, +0.0098] | 차이 관측됨 |
| 구조 차이 (D→A) | **+0.1979** | [+0.1886, +0.2071] | 차이 관측됨 |

### 4.3 임계값 분석 (§8)

| 모델 | Youden J | 부트스트랩 95% CI | 폭 | MCC(0.5) | MCC(J) | 판정 |
|---|---|---|---|---|---|---|
| A | 0.593 | [0.543, 0.653] | 0.110 | 0.8251 | 0.8342 | 안정 |
| B | 0.480 | [0.470, 0.570] | 0.100 | 0.8343 | 0.8350 | 안정 |
| **C** | 0.632 | [0.449, 0.712] | **0.262** | 0.8308 | 0.8363 | **불안정 → 0.5 고정** |
| D | 0.445 | [0.380, 0.447] | 0.067 | 0.6227 | 0.6261 | 안정 |

프로토콜 §8의 사전 규칙(CI 폭 > 0.2)에 따라 **C만 0.5로 회귀**한다. **Test의 사전확률이 78.8%로 Validation(50%)과 다르므로(§11.2.2), 여기서 선택한 임계값이 공식 Test에서 최적이라는 보장은 없다.**

---

## 5. 결과 해석 시 유의사항

### 5.1 스태킹이 성능을 낮춘 것은 예상 범위다

B→C에서 MCC가 0.0017 하락했으나 **CI가 0을 포함하여 판정은 유보**된다(§12 규칙). 효과 크기 자체는 앙상블 이론과 부합한다.

- Caruana et al.(2004)은 앙상블 선택의 이득이 **배깅/부스팅 이득의 약 1/2.5**에 불과하다고 보고했다. RF는 이미 배깅 기반이므로 추가 이득의 여지가 작다.
- 본 구성의 베이스 3개는 **시드만 다른 동일 RF**로 예측 상관이 높다. Kuncheva & Whitaker(2003)가 지적한 "다양성 지표와 성능의 무관계" 논의와 연결되는 사례다.

### 5.2 구조 차이를 "순수 구조 효과"로 부르지 않는다

D→A 차이(+0.2096)는 크지만, 프로토콜 H3의 규정대로 **"지정된 학습 조건 하의 파이프라인 차이"**로 서술한다. A와 D가 모두 미튜닝이라 조건은 대칭이나, 특징 스케일 처리(D만 표준화) 등 파이프라인 차이가 남아 있다.

### 5.3 튜닝 효과의 실질적 크기

A→B의 +0.0067은 통계적으로 유의하나, 프로토콜 §10.2의 허용 한도(MCC 절대 0.05)의 **약 1/7 수준**이다. 실질적 의미는 제한적이라고 서술해야 한다.

### 5.4 H2 검정력 축소

중복 제거로 교란 대상이 9개에서 **7개**로 줄었다(`n_uniq_child`, `n_uniq_cguid` 탈락). 탈락한 둘은 "쉬움" 분류였고 잔존한 `n_events`는 교란 제외 항목이라, 결정론적 규칙의 결과가 H2의 검정력을 약화시켰다. **규칙을 사후 변경하지 않으며** 이를 한계로 기록한다.

---

## 6. 산출 파일

| 파일 | 내용 |
|---|---|
| `df.pkl` | 원본 데이터프레임 |
| `h5_idx.npy` / `offtest_idx.npy` | **봉인 셋 인덱스** — Phase 6까지 로드 금지 |
| `featureset.json` | 전체/제거/최종 특징 목록, 중복 쌍과 ρ |
| `tr_idx.npy` / `va_idx.npy` | 학습·검증 분할 인덱스 |
| `old_run/` | 구 분할(Train 30,000) 실행 결과 보존 |
| `bo_state.pkl` | BO 중간 상태 (재개용) |
| `bo_result.json` | BO 최적 파라미터, 30회 이력, 알고리즘 표기 |
| `results_all.json` | 모델×시드별 전체 지표 |
| `prob_{model}_{seed}.npy` | Validation 예측 확률 (Phase 4 교란·MIA 실험 입력) |
