"""Phase 3 - 모델 B/C/D 학습 (재개 가능). Test 미사용.
사용: python3 run_models.py <model> [seed]
"""
import pandas as pd, numpy as np, json, os, sys, time, warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (matthews_corrcoef, f1_score, precision_score,
                             recall_score, average_precision_score, confusion_matrix)

OUT = '/home/claude/phase3/'
SEEDS = [42, 7, 123, 2024, 777]
RES = OUT + 'results_all.json'

df = pd.read_pickle(OUT + 'df.pkl')
fs = json.load(open(OUT + 'featureset.json')); FEAT = fs['features_final']
tr = np.load(OUT + 'tr_idx.npy'); va = np.load(OUT + 'va_idx.npy')
y = (df['label'] == 'malicious').astype(int).values
Xtr = df.loc[tr, FEAT].values; ytr = y[tr]
Xva = df.loc[va, FEAT].values; yva = y[va]
BP = json.load(open(OUT + 'bo_result.json'))['best_params']

def evaluate(yt, yp, yprob):
    tn, fp, fn, tp = confusion_matrix(yt, yp).ravel()
    return dict(MCC=matthews_corrcoef(yt, yp), F1=f1_score(yt, yp),
                Precision=precision_score(yt, yp), Recall=recall_score(yt, yp),
                PR_AUC=average_precision_score(yt, yprob), FPR=fp/(fp+tn),
                TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp))

res = json.load(open(RES)) if os.path.exists(RES) else {}

def save(model, seed, r, prob):
    res.setdefault(model, {})[str(seed)] = r
    json.dump(res, open(RES, 'w'), indent=2)
    np.save(OUT + f'prob_{model}_{seed}.npy', prob)

model = sys.argv[1]
seeds = [int(sys.argv[2])] if len(sys.argv) > 2 else SEEDS

for s in seeds:
    if str(s) in res.get(model, {}):
        print(f'{model} seed={s} 이미 완료 — 건너뜀'); continue
    t0 = time.time()

    if model == 'A_RF':
        m = RandomForestClassifier(n_estimators=300, random_state=s, n_jobs=-1)
        m.fit(Xtr, ytr); prob = m.predict_proba(Xva)[:, 1]

    elif model == 'B_RF_BO':
        m = RandomForestClassifier(**BP, random_state=s, n_jobs=-1)
        m.fit(Xtr, ytr); prob = m.predict_proba(Xva)[:, 1]

    elif model == 'D_MLP':
        sc = StandardScaler().fit(Xtr)
        m = MLPClassifier(max_iter=200, early_stopping=True,
                          n_iter_no_change=10, random_state=s)   # sklearn 기본 (100,)
        m.fit(sc.transform(Xtr), ytr); prob = m.predict_proba(sc.transform(Xva))[:, 1]

    elif model == 'C_STACK':
        bases = [RandomForestClassifier(**BP, random_state=s + k, n_jobs=-1) for k in range(3)]
        bases.append(HistGradientBoostingClassifier(random_state=s))
        oof_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        # OOF 예측으로 메타 학습 (§4.2.1 in-sample 예측 금지)
        oof = np.column_stack([
            cross_val_predict(b, Xtr, ytr, cv=oof_cv, method='predict_proba', n_jobs=1)[:, 1]
            for b in bases])
        meta = LogisticRegression(max_iter=1000).fit(oof, ytr)
        for b in bases: b.fit(Xtr, ytr)
        prob = meta.predict_proba(
            np.column_stack([b.predict_proba(Xva)[:, 1] for b in bases]))[:, 1]
    else:
        raise SystemExit('unknown model')

    r = evaluate(yva, (prob >= 0.5).astype(int), prob)
    r['seed'] = s; r['fit_s'] = round(time.time() - t0, 1)
    save(model, s, r, prob)
    print(f'{model} seed={s} MCC={r["MCC"]:.4f} ({r["fit_s"]}s)', flush=True)

