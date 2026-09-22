"""Phase 3 - 재개 가능한 GP-BO 실행기.
한 번 호출할 때 최대 N회 평가하고 상태를 디스크에 저장. 완료될 때까지 반복 호출.
"""
import pandas as pd, numpy as np, json, os, sys, time, pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import matthews_corrcoef
sys.path.insert(0, '/home/claude/phase3')
from gpbo import GPBayesOpt

OUT = '/home/claude/phase3/'
STATE = OUT + 'bo_state.pkl'
N_PER_CALL = int(sys.argv[1]) if len(sys.argv) > 1 else 12
N_INIT, N_ITER = 10, 20

df = pd.read_pickle(OUT + 'df.pkl')
fs = json.load(open(OUT + 'featureset.json')); FEAT = fs['features_final']
tr = np.load(OUT + 'tr_idx.npy')
y = (df['label'] == 'malicious').astype(int).values
Xtr = df.loc[tr, FEAT].values; ytr = y[tr]

SPACE = {
    'n_estimators':     ('int', 10, 200),
    'max_depth':        ('int', 5, 50),
    'max_features':     ('int', 1, len(FEAT)),
    'min_samples_split':('int', 2, 11),
    'min_samples_leaf': ('int', 1, 11),
    'criterion':        ('cat', ['gini', 'entropy']),
}
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

def objective(p):
    scores = []
    for itr, iva in cv.split(Xtr, ytr):
        m = RandomForestClassifier(**p, random_state=42, n_jobs=-1)
        m.fit(Xtr[itr], ytr[itr])
        scores.append(matthews_corrcoef(ytr[iva], m.predict(Xtr[iva])))
    return float(np.mean(scores))

opt = GPBayesOpt(SPACE, n_init=N_INIT, n_iter=N_ITER, seed=42)
if os.path.exists(STATE):
    st = pickle.load(open(STATE, 'rb'))
    opt.X, opt.y, opt.rng = st['X'], st['y'], st['rng']
    print(f'재개: 기존 {len(opt.y)}회 평가됨', flush=True)

TOTAL = N_INIT + N_ITER
done = 0
while len(opt.y) < TOTAL and done < N_PER_CALL:
    t0 = time.time()
    if len(opt.y) < N_INIT:
        u = opt._sample_u(1)[0]; tag = f'init {len(opt.y)+1}/{N_INIT}'
    else:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
        import warnings; warnings.filterwarnings('ignore')
        kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(len(SPACE)), nu=2.5) + WhiteKernel(1e-4)
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                      n_restarts_optimizer=0,
                                      random_state=opt.rng.randint(10**6))
        gp.fit(np.array(opt.X), np.array(opt.y))
        cand = opt._sample_u(2000)
        ei = opt._ei(gp, cand, best=max(opt.y))
        u = cand[int(np.argmax(ei))]
        tag = f'bo {len(opt.y)-N_INIT+1}/{N_ITER}'
    p = opt._decode(u)
    s = objective(p)
    opt.X.append(u); opt.y.append(s); done += 1
    print(f'  [{tag}] MCC={s:.4f} best={max(opt.y):.4f} ({time.time()-t0:.0f}s) {p}', flush=True)
    pickle.dump({'X': opt.X, 'y': opt.y, 'rng': opt.rng}, open(STATE, 'wb'))

if len(opt.y) >= TOTAL:
    b = int(np.argmax(opt.y))
    best = opt._decode(opt.X[b])
    json.dump({'best_params': best, 'best_cv_mcc': float(opt.y[b]),
               'history': [float(v) for v in opt.y], 'algorithm': 'GP-EI (Matern 2.5)'},
              open(OUT + 'bo_result.json', 'w'), indent=2)
    print(f'\n== BO 완료 == 최적 {best} CV MCC={opt.y[b]:.4f}', flush=True)
else:
    print(f'\n진행 {len(opt.y)}/{TOTAL} — 다시 호출하세요', flush=True)
