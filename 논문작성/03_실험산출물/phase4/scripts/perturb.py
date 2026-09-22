"""Phase 4 - 특징 교란 민감도 + 순열 중요도 (프로토콜 v2.9 §6, §6.4)
- 교란 대상 7개 (§6.3 "쉬움" 중 중복제거 후 잔존)
- 감소 단계 5/10/15/20/30%, flip rate는 고정 임계값 0.5 (§6.1)
- 분모: 원래 정확히 악성으로 분류한 표본만
- 유효성: 비음수 / Train 관측범위 / n_events 정합성 재계산 (§6.2, §6.3)
- 순열 중요도: Validation, scoring=MCC, n_repeats=10 (§6.4)
- H2: 중요도 vs 교란 민감도 S_i = -Δp/Δreduction 의 Spearman 상관
"""
import pandas as pd, numpy as np, json, os, warnings, sys
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import matthews_corrcoef, make_scorer
from scipy.stats import spearmanr

P3 = '/home/claude/phase3/'; OUT = '/home/claude/phase4/'
SEED = 42
STEPS = [0.05, 0.10, 0.15, 0.20, 0.30]
# §6.3 교란 대상: "쉬움" 9개 중 중복제거(n_uniq_child, n_uniq_cguid 탈락) 후 7개
PERTURB = ['net_connect', 'file_create', 'reg_write', 'reg_create_key',
           'reg_delete_key', 'reg_delete_value', 'duration_s']
EVENT9 = ['process_create', 'process_term', 'image_load', 'net_connect', 'file_create',
          'reg_create_key', 'reg_write', 'reg_delete_key', 'reg_delete_value']

df = pd.read_pickle(P3 + 'df.pkl')
FEAT = json.load(open(P3 + 'featureset.json'))['features_final']
tr = np.load(P3 + 'tr_idx.npy'); va = np.load(P3 + 'va_idx.npy')
y = (df['label'] == 'malicious').astype(int).values
BP = json.load(open(P3 + 'bo_result.json'))['best_params']
Xtr, ytr = df.loc[tr, FEAT].values, y[tr]
Xva, yva = df.loc[va, FEAT].values, y[va]
COL = {c: i for i, c in enumerate(FEAT)}
tr_min, tr_max = Xtr.min(0), Xtr.max(0)      # Train 관측범위 (§6.2 검증가능 조건)

sc = StandardScaler().fit(Xtr)

class Wrap(ClassifierMixin, BaseEstimator):
    """모델별 predict_proba를 통일 인터페이스로 감싼다 (sklearn estimator 호환)"""
    def __init__(self, kind=None, obj=None):
        self.kind, self.obj = kind, obj
        self.classes_ = np.array([0, 1])
        self.is_fitted_ = True
    def predict_proba(self, X):
        if self.kind == 'mlp': return self.obj.predict_proba(sc.transform(X))
        if self.kind == 'stack':
            bases, meta = self.obj
            return meta.predict_proba(np.column_stack([b.predict_proba(X)[:, 1] for b in bases]))
        return self.obj.predict_proba(X)
    def predict(self, X): return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
    def fit(self, X, y): return self          # 이미 학습된 모델을 감싸므로 no-op
    def __sklearn_is_fitted__(self): return True

def build(model):
    if model == 'A_RF':
        return Wrap('tree', RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1).fit(Xtr, ytr))
    if model == 'B_RF_BO':
        return Wrap('tree', RandomForestClassifier(**BP, random_state=SEED, n_jobs=-1).fit(Xtr, ytr))
    if model == 'D_MLP':
        return Wrap('mlp', MLPClassifier(max_iter=200, early_stopping=True,
                                         n_iter_no_change=10, random_state=SEED).fit(sc.transform(Xtr), ytr))
    bases = [RandomForestClassifier(**BP, random_state=SEED + k, n_jobs=-1) for k in range(3)]
    bases.append(HistGradientBoostingClassifier(random_state=SEED))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.column_stack([cross_val_predict(b, Xtr, ytr, cv=cv, method='predict_proba', n_jobs=1)[:, 1]
                           for b in bases])
    meta = LogisticRegression(max_iter=1000).fit(oof, ytr)
    for b in bases: b.fit(Xtr, ytr)
    return Wrap('stack', (bases, meta))

model = sys.argv[1]
print(f'=== {model} ===', flush=True)
m = build(model)

# ---------- 기준선: 원래 정확히 악성으로 분류한 표본 (§6.1 분모) ----------
p0 = m.predict_proba(Xva)[:, 1]
base_mask = (yva == 1) & (p0 >= 0.5)
Xb = Xva[base_mask]; pb = p0[base_mask]
print(f'분모(정확히 악성 분류): {base_mask.sum()} / 악성 {int((yva==1).sum())}', flush=True)

# ---------- 교란 민감도 ----------
sens = {}
for f in PERTURB:
    j = COL[f]
    rows = []
    for s in STEPS:
        Xp = Xb.copy()
        Xp[:, j] = Xp[:, j] * (1 - s)
        # 유효성 검사 (§6.2 검증가능 조건)
        valid = (Xp[:, j] >= 0) & (Xp[:, j] >= tr_min[j]) & (Xp[:, j] <= tr_max[j])
        # n_events 정합성 재계산 (§6.3)
        if f in EVENT9 and 'n_events' in COL:
            Xp[:, COL['n_events']] = Xb[:, COL['n_events']] - (Xb[:, j] - Xp[:, j])
            valid &= (Xp[:, COL['n_events']] >= 0)
        pp = m.predict_proba(Xp)[:, 1]
        v = valid
        rows.append(dict(step=s, n_valid=int(v.sum()), n_invalid=int((~v).sum()),
                         flip_rate=float((pp[v] < 0.5).mean()) if v.sum() else float('nan'),
                         mean_prob_drop=float((pb[v] - pp[v]).mean()) if v.sum() else float('nan')))
    # S_i = -Δp/Δreduction  (§6.4 부호 정의: 클수록 민감)
    xs = np.array([r['step'] for r in rows]); ys = np.array([r['mean_prob_drop'] for r in rows])
    S = float(np.polyfit(xs, ys, 1)[0])      # 확률 감소폭의 기울기 = 민감도
    sens[f] = dict(steps=rows, S_i=S)
    print(f'  {f:20s} S_i={S:+.4f}  flip@20%={rows[3]["flip_rate"]:.4f} '
          f'(무효 {rows[3]["n_invalid"]})', flush=True)

# ---------- 순열 중요도 (§6.4: Validation, MCC, n_repeats=10) ----------
print('  순열 중요도 계산 중...', flush=True)
pi = permutation_importance(m, Xva, yva, scoring=make_scorer(matthews_corrcoef),
                            n_repeats=10, random_state=SEED, n_jobs=1)
imp = {c: dict(mean=float(pi.importances_mean[i]), std=float(pi.importances_std[i]))
       for i, c in enumerate(FEAT)}

# ---------- H2: 중요도 vs 민감도 상관 ----------
xs = np.array([imp[f]['mean'] for f in PERTURB])
ys = np.array([sens[f]['S_i'] for f in PERTURB])
rho, pval = spearmanr(xs, ys)

res = dict(n_denominator=int(base_mask.sum()), sensitivity=sens, importance=imp,
           H2=dict(n=len(PERTURB), spearman_rho=float(rho), p_value=float(pval),
                   features=PERTURB))
path = OUT + 'perturb.json'
allr = json.load(open(path)) if os.path.exists(path) else {}
allr[model] = res
json.dump(allr, open(path, 'w'), indent=2)

print(f'\n  [H2] n={len(PERTURB)}  Spearman rho={rho:+.4f}  p={pval:.4f}', flush=True)
print('  순열 중요도 상위 5:', flush=True)
for c, v in sorted(imp.items(), key=lambda x: -x[1]['mean'])[:5]:
    print(f'    {c:20s} {v["mean"]:+.4f} ± {v["std"]:.4f}', flush=True)
