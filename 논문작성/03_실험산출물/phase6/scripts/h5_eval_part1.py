"""Phase 6b-1 - H5-Test 봉인 해제 및 평가 (1회 한정)
프로토콜 v3.4 §10.4. 탐지·지연·강건성 축. (MIA는 6b-2에서 고정 감사셋으로 별도 수행)
봉인 해제 전 해시 검증 -> 불일치 시 즉시 중단.
"""
import pandas as pd, numpy as np, json, hashlib, time, sys, warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef, confusion_matrix

P3 = '/home/claude/phase3/'; OUT = '/home/claude/phase6/'
SEED = 42; N_BOOT = 2000
EXPECTED_HASH = '47254fd15bb1f192'

# ================= 1. 봉인 해제 전 무결성 검증 =================
h5 = np.load(P3 + 'h5_idx.npy')
got = hashlib.sha256(np.sort(np.asarray(h5)).tobytes()).hexdigest()[:16]
print(f'H5-Test 해시 검증: 기대 {EXPECTED_HASH} / 실제 {got}  ->  '
      f'{"일치 ✅ 봉인 해제" if got == EXPECTED_HASH else "불일치 ❌"}', flush=True)
if got != EXPECTED_HASH: sys.exit('해시 불일치 — 평가 중단')

cfg23 = json.load(open(OUT + 'step23.json')); cfg4 = json.load(open(OUT + 'step4_perturb.json'))
N_EST, MSL, DEC = cfg23['n_estimators'], cfg23['min_samples_leaf'], cfg23['decimals']
RATIO = cfg4['chosen_ratio']
PERTURB = ['net_connect', 'file_create', 'reg_write', 'reg_create_key',
           'reg_delete_key', 'reg_delete_value', 'duration_s']
EVENT9 = ['process_create', 'process_term', 'image_load', 'net_connect', 'file_create',
          'reg_create_key', 'reg_write', 'reg_delete_key', 'reg_delete_value']

df = pd.read_pickle(P3 + 'df.pkl')
FEAT = json.load(open(P3 + 'featureset.json'))['features_final']
tr = np.load(P3 + 'tr_idx.npy')
y = (df['label'] == 'malicious').astype(int).values
Xtr, ytr = df.loc[tr, FEAT].values, y[tr]
Xh, yh = df.loc[h5, FEAT].values, y[h5]
COL = {c: i for i, c in enumerate(FEAT)}
tr_min, tr_max = Xtr.min(0), Xtr.max(0)
print(f'H5-Test: {len(yh)}건, 악성 {yh.mean()*100:.1f}%\n', flush=True)

# ================= 2. 세 모델 학습 (Train만, 선택 단계에서 확정된 설정) =================
def perturb(Xs, f, s=0.20):
    j = COL[f]; Xp = Xs.copy(); Xp[:, j] = Xp[:, j] * (1 - s)
    v = (Xp[:, j] >= 0) & (Xp[:, j] >= tr_min[j]) & (Xp[:, j] <= tr_max[j])
    if f in EVENT9:
        Xp[:, COL['n_events']] = Xs[:, COL['n_events']] - (Xs[:, j] - Xp[:, j])
        v &= Xp[:, COL['n_events']] >= 0
    return Xp, v

M = {}
M['A_none'] = (RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1).fit(Xtr, ytr), None)
M['Selective'] = (RandomForestClassifier(n_estimators=N_EST, random_state=SEED, n_jobs=-1).fit(Xtr, ytr), None)

# 일괄: 경량화 + 정규화 + 이산화 + 교란학습 (step4와 동일한 절차로 재구성)
m0 = RandomForestClassifier(n_estimators=N_EST, min_samples_leaf=MSL, random_state=SEED, n_jobs=-1).fit(Xtr, ytr)
p0 = np.round(m0.predict_proba(Xtr)[:, 1], DEC); tb = (ytr == 1) & (p0 >= 0.5)
pool = []
for f in PERTURB:
    Xp, v = perturb(Xtr[tb], f)
    pool.append(Xp[v & (np.round(m0.predict_proba(Xp)[:, 1], DEC) < 0.5)])
pool = np.vstack(pool)
rs = np.random.RandomState(SEED)
for r in [0.10, 0.20, 0.30, 0.40, 0.50]:          # step4와 동일한 난수 소비 순서 재현
    n_aug = int(round(int((ytr == 1).sum()) * r))
    aug = pool[rs.choice(len(pool), n_aug, replace=n_aug > len(pool))]
    if r == RATIO: AUG = aug
mb = RandomForestClassifier(n_estimators=N_EST, min_samples_leaf=MSL, random_state=SEED, n_jobs=-1).fit(
    np.vstack([Xtr, AUG]), np.concatenate([ytr, np.ones(len(AUG), dtype=int)]))
M['Batch'] = (mb, DEC)
print(f'일괄 모델 재구성: flip 풀 {len(pool)}건, 증강 {len(AUG)}건 (step4 기록 {cfg4["pool_size"]}건과 대조)', flush=True)

def prob(name, X):
    m, d = M[name]; p = m.predict_proba(X)[:, 1]
    return p if d is None else np.round(p, d)

# ================= 3. 표본 단위 지표 (부트스트랩용 원자료) =================
P = {k: prob(k, Xh) for k in M}
PRED = {k: (P[k] >= 0.5).astype(int) for k in M}

def flip_matrix(name):
    """표본별·특징별 flip 여부. 정확히 악성 분류된 표본만 분모(§6.1)"""
    base = (yh == 1) & (PRED[name] == 1)
    F = np.full((len(yh), len(PERTURB)), np.nan)
    for j, f in enumerate(PERTURB):
        Xp, v = perturb(Xh, f)
        fl = (prob(name, Xp) < 0.5).astype(float)
        F[:, j] = np.where(base & v, fl, np.nan)
    return F
FLIP = {k: flip_matrix(k) for k in M}

def max_flip_rate(F, idx):
    sub = F[idx]; rates = np.nanmean(sub, axis=0)
    return float(np.nanmax(rates))

# ================= 4. 지연시간 (batch=10, 반복 측정) =================
LAT = {}
Xb = Xh[:10]
for k in M:
    for _ in range(20): prob(k, Xb)
    ts = np.empty(500)
    for i in range(500):
        t0 = time.perf_counter(); prob(k, Xb); ts[i] = time.perf_counter() - t0
    LAT[k] = ts * 1000 / 10
    print(f'  {k:10s} 지연 p50 {np.median(LAT[k]):.4f} ms/샘플', flush=True)

# ================= 5. 점추정 =================
def point(k):
    tn, fp, fn, tp = confusion_matrix(yh, PRED[k]).ravel()
    return dict(MCC=float(matthews_corrcoef(yh, PRED[k])), FPR=float(fp / (fp + tn)),
                max_flip=max_flip_rate(FLIP[k], np.arange(len(yh))),
                lat_p50=float(np.median(LAT[k])), TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp))
PT = {k: point(k) for k in M}
print('\n=== H5-Test 점추정 ===')
for k in M:
    p = PT[k]
    print(f"  {k:10s} MCC={p['MCC']:.4f} FPR={p['FPR']:.4f} 최대flip={p['max_flip']:.4f} 지연={p['lat_p50']:.4f}ms")

# ================= 6. 대응 부트스트랩: Selective - Batch =================
rng = np.random.RandomState(SEED)
n = len(yh)
dMCC, dFLIP, dLAT = [], [], []
for _ in range(N_BOOT):
    i = rng.randint(0, n, n)
    dMCC.append(matthews_corrcoef(yh[i], PRED['Selective'][i]) - matthews_corrcoef(yh[i], PRED['Batch'][i]))
    dFLIP.append(max_flip_rate(FLIP['Selective'], i) - max_flip_rate(FLIP['Batch'], i))
    ls = rng.choice(LAT['Selective'], len(LAT['Selective'])); lb = rng.choice(LAT['Batch'], len(LAT['Batch']))
    dLAT.append((np.median(ls) - np.median(lb)) / np.median(lb))     # 상대 차이
def ci(a): return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

res = dict(hash_verified=True, n_h5=int(n), config=dict(n_est=N_EST, msl=MSL, dec=DEC, ratio=RATIO),
           point=PT,
           diff_sel_minus_batch=dict(
               MCC=dict(est=PT['Selective']['MCC'] - PT['Batch']['MCC'], ci=ci(dMCC)),
               max_flip=dict(est=PT['Selective']['max_flip'] - PT['Batch']['max_flip'], ci=ci(dFLIP)),
               latency_rel=dict(est=(PT['Selective']['lat_p50'] - PT['Batch']['lat_p50']) / PT['Batch']['lat_p50'],
                                ci=ci(dLAT))))
json.dump(res, open(OUT + 'h5_eval_part1.json', 'w'), indent=2)
np.save(OUT + 'h5_pred_probs.npy', np.vstack([P[k] for k in ['A_none', 'Selective', 'Batch']]))

print('\n=== 차이(Selective − Batch) 95% 부트스트랩 CI ===')
for k, v in res['diff_sel_minus_batch'].items():
    print(f"  {k:12s} Δ={v['est']:+.4f}  CI [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}]")
