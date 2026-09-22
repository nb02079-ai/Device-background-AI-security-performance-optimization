# 모델 B·C 구현 문서 — 베이지안 최적화와 스태킹

**작성일**: 2026-09-20
**대상 프로토콜**: v2.9
**데이터**: DMBD 2025, Train 24,000 / Validation 8,000
**목적**: 모델 B(RF + 베이지안 최적화)와 C(스태킹)의 구현을 원리·코드·결과 순으로 기록한다.

> 모델 A(RF 기본)·D(MLP)는 `phase3_model_implementation.md` 참고. 본 문서는 **튜닝 방법론 축**을 담당하는 두 모델에 집중한다.

---

# 제1부. 모델 B — RandomForest + 베이지안 최적화

## 1. 왜 베이지안 최적화인가

하이퍼파라미터를 고르는 방법은 셋이다.

| 방법 | 원리 | 한계 |
|---|---|---|
| 그리드 서치 | 격자의 모든 조합 시도 | 차원이 늘면 조합 수가 폭발 |
| 랜덤 서치 | 무작위 조합 시도 | **과거 시도에서 배우지 않음** |
| **베이지안 최적화** | 과거 결과로 대리 모델(surrogate)을 세우고 다음 시도를 추론 | 순차적이라 병렬화 제약 |

핵심 전제는 **평가 1회가 비싸다**는 것이다. 본 연구에서 목적함수 1회 평가는 Train 24,000건에 대한 3-fold 교차검증이며, 최대 24초가 걸렸다. 30회면 예산이 소진되므로 시행을 낭비할 수 없다.

## 2. 알고리즘 — GP + Expected Improvement

프로토콜 §4.2는 당초 hyperopt의 TPE를 지정했으나, 실행 환경에 해당 패키지가 없고 네트워크가 차단되어 **GP 기반 BO를 직접 구현**했다. 탐색 공간·시행 횟수·목적함수는 변경하지 않았다.

### 2.1 동작 원리

```
1. 초기 무작위 10회 평가  →  (x, y) 관측점 확보
2. 반복 20회:
     a. 관측점에 가우시안 프로세스(GP)를 적합
        → 미평가 지점의 성능 예측값 μ(x)와 불확실성 σ(x)를 얻음
     b. 획득함수 EI(x)를 최대화하는 지점을 선택
     c. 그 지점을 실제 평가하고 관측점에 추가
```

**Expected Improvement**: 현재 최고값 `f*`에 대해

```
EI(x) = (μ(x) − f* − ξ)·Φ(z) + σ(x)·φ(z),   z = (μ(x) − f* − ξ) / σ(x)
```

- 첫 항은 **개선 기대값** — 예측 성능이 최고값보다 얼마나 높은가
- 둘째 항은 **불확실성 보너스** — 아직 안 가본 영역에 가중
- `ξ = 0.01`은 탐색(exploration) 가중치

이 두 항의 균형이 BO의 핵심이다. 첫 항만 보면 국소 최적에 갇히고, 둘째 항만 보면 무작위 탐색이 된다.

**커널**: Matérn(ν=2.5) × 상수 + 백색잡음. Matérn 2.5는 RBF보다 덜 매끄러운 함수를 허용해 하이퍼파라미터 응답면처럼 거친 표면에 적합하다.

### 2.2 혼합 탐색공간 처리

RF의 하이퍼파라미터는 정수형(`n_estimators`)과 범주형(`criterion`)이 섞여 있는데, GP는 연속 공간에서 동작한다. 모든 차원을 **[0,1]로 정규화**한 뒤 디코딩하는 방식으로 해결했다.

```python
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
```

## 3. 구현 — `gpbo.py`

```python
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel


class GPBayesOpt:
    """연속/정수/범주 혼합 탐색공간에 대한 GP-EI 베이지안 최적화."""

    def __init__(self, space, n_init=10, n_iter=20, seed=42):
        self.space = space
        self.names = list(space.keys())
        self.n_init = n_init
        self.n_iter = n_iter
        self.rng = np.random.RandomState(seed)
        self.X, self.y = [], []

    def _sample_u(self, n):
        return self.rng.rand(n, len(self.names))

    def _ei(self, gp, U, best, xi=0.01):
        mu, sd = gp.predict(U, return_std=True)
        sd = np.maximum(sd, 1e-9)          # 0 나눗셈 방지
        imp = mu - best - xi
        z = imp / sd
        return imp * norm.cdf(z) + sd * norm.pdf(z)
```

반복부 핵심 (`run_bo.py`):

```python
kernel = (ConstantKernel(1.0)
          * Matern(length_scale=np.ones(len(SPACE)), nu=2.5)
          + WhiteKernel(1e-4))
gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                              n_restarts_optimizer=0,
                              random_state=opt.rng.randint(10**6))
gp.fit(np.array(opt.X), np.array(opt.y))

cand = opt._sample_u(2000)                     # 후보 2,000개 무작위 생성
ei = opt._ei(gp, cand, best=max(opt.y))        # 각 후보의 EI 계산
u = cand[int(np.argmax(ei))]                   # EI 최대 지점 선택
```

> **EI 최대화를 무작위 후보로 처리한 이유**: 획득함수 자체를 경사법으로 최적화할 수도 있으나, 6차원 공간에서 후보 2,000개 평가는 GP 예측 한 번이면 충분히 빠르고 국소최적에 빠지지 않는다.

### 3.1 재개 가능 설계

실행 환경의 시간 제약(호출당 300초)으로 30회를 한 번에 돌릴 수 없어, 매 평가 후 상태를 저장하고 재호출 시 이어가도록 구현했다.

```python
if os.path.exists(STATE):
    st = pickle.load(open(STATE, 'rb'))
    opt.X, opt.y, opt.rng = st['X'], st['y'], st['rng']   # 난수 상태까지 복원
...
pickle.dump({'X': opt.X, 'y': opt.y, 'rng': opt.rng}, open(STATE, 'wb'))
```

**난수 상태(RNG)까지 보존**하므로 중단·재개 여부가 결과에 영향을 주지 않는다. 이는 재현성 확보를 위한 필수 조치다.

## 4. 탐색 공간과 목적함수

```python
SPACE = {
    'n_estimators':     ('int', 10, 200),
    'max_depth':        ('int', 5, 50),
    'max_features':     ('int', 1, 12),      # 실제 특징 수에 맞춰 상한 조정
    'min_samples_split':('int', 2, 11),
    'min_samples_leaf': ('int', 1, 11),
    'criterion':        ('cat', ['gini', 'entropy']),
}
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

def objective(p):
    """Train 내부 3-fold CV의 MCC (§8 주지표와 일치, Test 미사용)"""
    scores = []
    for itr, iva in cv.split(Xtr, ytr):
        m = RandomForestClassifier(**p, random_state=42, n_jobs=-1)
        m.fit(Xtr[itr], ytr[itr])
        scores.append(matthews_corrcoef(ytr[iva], m.predict(Xtr[iva])))
    return float(np.mean(scores))
```

> **선행 구현과의 결정적 차이**: IDS-ML(`MTH_IDS_IoTJ.py` L453-455)은 목적함수를 **Test set 점수**로 계산한다. 이렇게 하면 Test가 더 이상 미관측 데이터가 아니게 되어 최종 성능이 과대평가된다. 본 구현은 **Train 내부 CV**만 사용한다(프로토콜 §4.2.1).

## 5. 탐색 결과 (2026-09-20 재실행)

| 단계 | 최고 CV MCC |
|---|---|
| 초기 무작위 10회 | 0.8245 |
| BO 20회 | **0.8285** (반복 8회차에 갱신) |

**최적 하이퍼파라미터**

```python
{'n_estimators': 54, 'max_depth': 50, 'max_features': 2,
 'min_samples_split': 11, 'min_samples_leaf': 1, 'criterion': 'gini'}
```

관측 CV MCC 범위는 0.7415~0.8285였다.

**최악값이 나온 조합의 공통점**: `max_features=1` 또는 `max_depth≤12`처럼 표현력이 과도하게 제한된 경우다. 예컨대 `max_features=1, min_samples_leaf=11` 조합은 0.7415로 최고값보다 0.087 낮았다.

**흥미로운 점**: 최적 조합의 `n_estimators=54`는 모델 A의 300보다 훨씬 작고, `max_features=2`는 12개 중 2개만 후보로 본다. **모델을 더 크게 만드는 방향이 아니라 무작위성을 키우는 방향**으로 최적화가 진행됐다 — Breiman이 RF에서 특징 무작위 선택으로 나무 간 상관을 낮춘 원리와 부합한다.

---

# 제2부. 모델 C — 스태킹 앙상블

## 6. 구조

```
입력 x (12 특징)
 ├→ RF (seed+0, B의 최적 파라미터) ──→ p₁ ┐
 ├→ RF (seed+1, 동일 파라미터)     ──→ p₂ ├→ 로지스틱 회귀 → 최종 확률
 ├→ RF (seed+2, 동일 파라미터)     ──→ p₃ │      (메타모델)
 └→ HistGradientBoosting           ──→ p₄ ┘
```

- **베이스**: RF 3종(시드만 다름) + HistGradientBoosting 1종
- **메타**: 로지스틱 회귀
- **메타 학습 데이터**: Train에 대한 **OOF(Out-Of-Fold) 예측** 5-fold

## 7. 핵심 구현 지점 — OOF

스태킹에서 가장 흔한 실수는 메타모델을 **베이스 모델의 in-sample 예측**으로 학습시키는 것이다.

**틀린 방법** (IDS-ML `Tree-based_IDS_GlobeCom19.py` L320-327의 실제 구현):

```python
p_train = base_model.predict(X_train)    # 자기가 학습한 데이터를 예측
meta_model.fit(p_train, y_train)
```

베이스 모델은 학습 데이터를 거의 완벽히 맞히므로, 메타모델은 **"베이스 모델은 항상 옳다"**는 비현실적 패턴을 학습한다. 배포 시 성능이 떨어진다.

**본 구현 (OOF)**:

```python
oof_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof = np.column_stack([
    cross_val_predict(b, Xtr, ytr, cv=oof_cv, method='predict_proba', n_jobs=1)[:, 1]
    for b in bases])
meta = LogisticRegression(max_iter=1000).fit(oof, ytr)
```

`cross_val_predict`는 각 샘플에 대해 **그 샘플이 학습에 포함되지 않은 fold의 모델**이 낸 예측을 반환한다. 따라서 메타모델이 보는 예측 품질이 실제 배포 시와 같아진다.

**예측 단계**:

```python
for b in bases:
    b.fit(Xtr, ytr)          # 전체 Train으로 재학습
prob = meta.predict_proba(
    np.column_stack([b.predict_proba(Xva)[:, 1] for b in bases]))[:, 1]
```

## 8. 설계 선택의 근거

**메타모델을 로지스틱 회귀로 둔 이유**

Caruana et al.(2004)의 관찰이 직접적 근거다 — **강하게 상관된 입력 모델이 많고 검증 데이터가 적으면 극적으로 과적합**한다. 본 구성은 베이스 3개가 시드만 다른 동일 RF라 예측 상관이 높다. 메타모델까지 복잡하면 2차 과적합이 발생한다.

**HistGradientBoosting을 쓴 이유**

당초 프로토콜은 `GradientBoostingClassifier`를 지정했으나, OOF 5-fold × 5시드 = 25회 학습이 필요해 연산 시간이 과도했다. 동일 알고리즘 계열의 히스토그램 기반 고속 구현으로 대체했다(프로토콜 §4.2, 변경이력 기록).

---

# 제3부. 결과와 해석

## 9. Validation 성능 (시드 5회 평균±표준편차)

| 모델 | MCC | F1 | Precision | Recall | PR-AUC | FPR | 학습시간/시드 |
|---|---|---|---|---|---|---|---|
| A: RF 기본 | 0.8246±0.0008 | 0.9117 | 0.9175 | 0.9060 | 0.9700 | 0.0814 | 6.3초 |
| **B: RF+BO** | **0.8342±0.0015** | 0.9154 | **0.9321** | 0.8992 | 0.9725 | **0.0655** | **0.9초** |
| C: RF+BO+스태킹 | 0.8325±0.0009 | 0.9148 | 0.9290 | 0.9011 | **0.9740** | 0.0689 | 15.0초 |

## 10. 차이의 95% 신뢰구간 (대응 비교)

| 비교 | Δ MCC | 95% CI | 판정 |
|---|---|---|---|
| 튜닝 효과 (A→B) | +0.0097 | [+0.0069, +0.0124] | 차이 관측됨 |
| **스태킹 효과 (B→C)** | −0.0017 | **[−0.0045, +0.0011]** | **유의미한 차이 발견 못함** |
| 튜닝+스태킹 (A→C) | +0.0080 | [+0.0061, +0.0098] | 차이 관측됨 |

## 11. 해석 시 유의사항

### 11.1 BO가 성능과 속도를 함께 개선했다

B는 A보다 MCC가 높으면서 **학습이 7배 빠르다**(0.9초 vs 6.3초). 최적 `n_estimators=54`가 고정값 300보다 훨씬 작기 때문이다. 오탐률도 0.0814→0.0655로 낮아졌다.

즉 **이 축에서는 탐지 성능과 자원 효율성이 함께 개선**되었다. H1("탐지 성능이 높다고 다른 축이 보장되지 않는다")의 반례가 되지 못하는 사례로 기록해야 한다.

### 11.2 스태킹 효과는 판정 유보다

구 분할(Train 30,000 / Val 10,000)에서는 CI가 [−0.0036, −0.0003]으로 "하락 확정"이었으나, 새 분할(Val 8,000)에서 CI가 넓어져 0을 포함하게 됐다.

**효과 크기 자체는 거의 변하지 않았다**(−0.0020 → −0.0017). 달라진 것은 표본 수에 따른 **검정력**이다. 프로토콜 §12 규칙에 따라 *"본 실험에서 통계적으로 유의미한 차이를 발견하지 못함"*으로 기록하며, **"차이가 없다"고 서술하지 않는다.**

### 11.3 스태킹이 이득을 내지 못한 것은 이론적으로 예상된 바다

Phase 0의 앙상블 이론 조사에서 사전에 기록한 두 근거가 있다.

- **Caruana et al.(2004)**: 앙상블 선택의 이득은 배깅/부스팅 이득의 약 1/2.5에 불과하다. RF는 이미 배깅 기반이므로 추가 이득의 여지가 작다.
- **Kuncheva & Whitaker(2003)**: 다양성 지표와 앙상블 성능 사이에 확립된 연관이 없다. 본 구성의 베이스 3개는 **시드만 다른 동일 RF**로 예측 상관이 높아, 다양성이 낮은 전형적 사례다.

결과가 나온 뒤 설명을 끌어온 것이 아니라 **사전에 세운 기대가 확인된 것**이므로 논의에서 그렇게 서술한다.

### 11.4 모델 C의 임계값은 0.5로 고정된다

| 모델 | Youden J | 부트스트랩 95% CI | 폭 | 판정 |
|---|---|---|---|---|
| A | 0.593 | [0.543, 0.653] | 0.110 | 안정 |
| B | 0.480 | [0.470, 0.570] | 0.100 | 안정 |
| **C** | 0.632 | **[0.449, 0.712]** | **0.262** | **불안정 → 0.5 고정** |
| D | 0.445 | [0.380, 0.447] | 0.067 | 안정 |

프로토콜 §8의 사전 규칙(CI 폭 > 0.2이면 불안정)에 따라 C만 0.5로 회귀한다. 스태킹 메타모델의 출력 분포가 특정 구간에 몰려 Youden 최적점이 흔들리는 것으로 보인다. **사전 규칙에 따른 자동 판정이며, 결과를 보고 정한 것이 아니다.**

---

## 12. 산출 파일

| 파일 | 내용 |
|---|---|
| `gpbo.py` | GP-EI 베이지안 최적화 클래스 |
| `run_bo.py` | 탐색공간·목적함수 정의, 재개 가능 실행기 |
| `bo_state.pkl` | BO 중간 상태 (X, y, RNG) |
| `bo_result.json` | 최적 파라미터, 30회 이력, 알고리즘 표기 |
| `run_models.py` | 모델 A/B/C/D 학습 (재개 가능) |
| `results_all.json` | 모델×시드별 전체 지표 |
| `prob_{model}_{seed}.npy` | Validation 예측 확률 (Phase 4 입력) |
| `old_run/` | 구 분할(Train 30,000) 실행 결과 보존 |
