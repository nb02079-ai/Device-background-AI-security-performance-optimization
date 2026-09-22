"""Phase 4 - 멤버십 추론 공격 (프로토콜 v2.9 §5)
옵션 B (family 구조 부재) + 5.3절 1순위 shadow-model 방식
- target: Phase 3 모델 A/B/C/D (Train 24,000으로 학습)
- member = Train, non-member = Validation, 연도 분포 매칭으로 교란 통제 (§5.2)
- 역할 3분리: 공격학습 / 개선안선택 / 최종감사 (§5.1)
- 지표: Accuracy, Precision, Recall, F1, ROC-AUC, AUC_adv(§5.4), TPR@1%FPR
공식 Test·H5-Test 미사용 (§9)
"""
import pandas as pd, numpy as np, json, os, warnings, sys
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
                             recall_score, f1_score, roc_curve)

P3 = '/home/claude/phase3/'; OUT = '/home/claude/phase4/'
os.makedirs(OUT, exist_ok=True)
SEED = 42
N_SHADOW = 5
rng = np.random.RandomState(SEED)

df = pd.read_pickle(P3 + 'df.pkl')
FEAT = json.load(open(P3 + 'featureset.json'))['features_final']
tr = np.load(P3 + 'tr_idx.npy'); va = np.load(P3 + 'va_idx.npy')
y = (df['label'] == 'malicious').astype(int).values
BP = json.load(open(P3 + 'bo_result.json'))['best_params']
Xall = df[FEAT].values
year = df['year'].fillna(-1).astype(int).values

Xtr, ytr = Xall[tr], y[tr]
sc = StandardScaler().fit(Xtr)

def build(model, Xt, yt, seed):
    """모델 종류별 학습 후 (predict_proba 함수) 반환"""
    if model == 'A_RF':
        m = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1).fit(Xt, yt)
        return lambda X: m.predict_proba(X)
    if model == 'B_RF_BO':
        m = RandomForestClassifier(**BP, random_state=seed, n_jobs=-1).fit(Xt, yt)
        return lambda X: m.predict_proba(X)
    if model == 'D_MLP':
        s2 = StandardScaler().fit(Xt)
        m = MLPClassifier(max_iter=200, early_stopping=True,
                          n_iter_no_change=10, random_state=seed).fit(s2.transform(Xt), yt)
        return lambda X: m.predict_proba(s2.transform(X))
    if model == 'C_STACK':
        bases = [RandomForestClassifier(**BP, random_state=seed + k, n_jobs=-1) for k in range(3)]
        bases.append(HistGradientBoostingClassifier(random_state=seed))
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        oof = np.column_stack([cross_val_predict(b, Xt, yt, cv=cv,
                               method='predict_proba', n_jobs=1)[:, 1] for b in bases])
        meta = LogisticRegression(max_iter=1000).fit(oof, yt)
        for b in bases: b.fit(Xt, yt)
        def f(X):
            p = meta.predict_proba(np.column_stack([b.predict_proba(X)[:, 1] for b in bases]))
            return p
        return f
    raise ValueError(model)


def attack_features(proba, ytrue):
    """공격자가 관찰 가능한 정보: 예측 확률 (§5.1 블랙박스 가정)
    확신도 + 정답 클래스 확률 + 엔트로피"""
    conf = proba.max(1)
    p_true = proba[np.arange(len(ytrue)), ytrue]
    eps = 1e-12
    ent = -(proba * np.log(proba + eps)).sum(1)
    return np.column_stack([conf, p_true, ent])


def match_year(idx_a, idx_b, rs):
    """연도 분포 매칭 (§5.2 DMBD 교란 통제): 연도별로 min(개수)만큼 각각 추출"""
    ya, yb = year[idx_a], year[idx_b]
    keep_a, keep_b = [], []
    for v in np.unique(np.concatenate([ya, yb])):
        aa, bb = idx_a[ya == v], idx_b[yb == v]
        n = min(len(aa), len(bb))
        if n == 0: continue
        keep_a.append(rs.choice(aa, n, replace=False))
        keep_b.append(rs.choice(bb, n, replace=False))
    return np.concatenate(keep_a), np.concatenate(keep_b)


def tpr_at_fpr(yt, sc_, target=0.01):
    fpr, tpr, _ = roc_curve(yt, sc_)
    return float(np.interp(target, fpr, tpr))


# ---------- member/non-member 구성 + 연도 매칭 ----------
mem_all, non_all = match_year(tr, va, rng)
print(f'연도 매칭 후: member {len(mem_all)} / non-member {len(non_all)}', flush=True)

# 역할 3분리 (§5.1): 공격학습 50% / 개선안선택 25% / 최종감사 25%
def split3(idx, rs):
    p = rs.permutation(len(idx)); idx = idx[p]
    n = len(idx); a = int(n*0.5); b = int(n*0.75)
    return idx[:a], idx[a:b], idx[b:]

mem_atk, mem_sel, mem_aud = split3(mem_all, rng)
non_atk, non_sel, non_aud = split3(non_all, rng)
print(f'  공격학습 {len(mem_atk)}+{len(non_atk)} / 개선안선택 {len(mem_sel)}+{len(non_sel)} / 최종감사 {len(mem_aud)}+{len(non_aud)}\n', flush=True)
np.save(OUT+'mia_audit_mem.npy', mem_aud); np.save(OUT+'mia_audit_non.npy', non_aud)

model = sys.argv[1]
print(f'=== {model} ===', flush=True)

# ---------- 1) Shadow model로 공격 모델 학습 (§5.3 1순위) ----------
Xsh_list, ysh_list = [], []
for k in range(N_SHADOW):
    perm = rng.permutation(len(tr))
    in_idx, out_idx = tr[perm[:12000]], tr[perm[12000:]]
    pf = build(model, Xall[in_idx], y[in_idx], seed=1000 + k)
    for idx, lab in [(in_idx, 1), (out_idx, 0)]:
        Xsh_list.append(attack_features(pf(Xall[idx]), y[idx]))
        ysh_list.append(np.full(len(idx), lab))
    print(f'  shadow {k+1}/{N_SHADOW} 완료', flush=True)

Xsh = np.vstack(Xsh_list); ysh = np.concatenate(ysh_list)
attacker = LogisticRegression(max_iter=1000).fit(Xsh, ysh)

# ---------- 2) target 모델에 공격 적용 ----------
target = build(model, Xtr, ytr, seed=SEED)

def eval_on(mem_idx, non_idx, tag):
    Xa = np.vstack([attack_features(target(Xall[mem_idx]), y[mem_idx]),
                    attack_features(target(Xall[non_idx]), y[non_idx])])
    ya = np.concatenate([np.ones(len(mem_idx)), np.zeros(len(non_idx))])
    s_ = attacker.predict_proba(Xa)[:, 1]
    pred = (s_ >= 0.5).astype(int)
    auc = roc_auc_score(ya, s_)
    return dict(tag=tag, n=len(ya),
                Accuracy=accuracy_score(ya, pred), Precision=precision_score(ya, pred, zero_division=0),
                Recall=recall_score(ya, pred), F1=f1_score(ya, pred, zero_division=0),
                ROC_AUC=float(auc), AUC_adv=float(max(auc, 1-auc)),
                TPR_at_1pct_FPR=tpr_at_fpr(ya, s_))

res = {'selection': eval_on(mem_sel, non_sel, 'selection'),
       'audit': eval_on(mem_aud, non_aud, 'audit')}

# 참고: 확신도 격차
c_mem = target(Xall[mem_aud]).max(1).mean(); c_non = target(Xall[non_aud]).max(1).mean()
res['confidence_gap'] = float(c_mem - c_non)

path = OUT + 'mia.json'
all_res = json.load(open(path)) if os.path.exists(path) else {}
all_res[model] = res
json.dump(all_res, open(path, 'w'), indent=2)

for t in ['selection', 'audit']:
    r = res[t]
    print(f"  [{t:9s}] AUC={r['ROC_AUC']:.4f} AUC_adv={r['AUC_adv']:.4f} "
          f"Acc={r['Accuracy']:.4f} TPR@1%FPR={r['TPR_at_1pct_FPR']:.4f}", flush=True)
print(f"  확신도 격차(member-nonmember) = {res['confidence_gap']:+.4f}", flush=True)
