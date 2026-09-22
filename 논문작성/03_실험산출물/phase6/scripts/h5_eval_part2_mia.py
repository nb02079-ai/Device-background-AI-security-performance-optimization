"""Phase 6b-2 - 최종 모델의 MIA 평가 (Phase 4 고정 감사셋 사용, §9)
각 모델의 학습 절차를 shadow model이 그대로 모사한다(일괄은 교란학습 포함).
감사셋 대응 부트스트랩으로 AUC_adv 차이(Selective - Batch)의 CI 산출.
"""
import pandas as pd, numpy as np, json, sys, warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

P3 = '/home/claude/phase3/'; P4 = '/home/claude/phase4/'; OUT = '/home/claude/phase6/'
SEED = 42; N_BOOT = 2000
cfg23 = json.load(open(OUT + 'step23.json')); cfg4 = json.load(open(OUT + 'step4_perturb.json'))
N_EST, MSL, DEC, RATIO = cfg23['n_estimators'], cfg23['min_samples_leaf'], cfg23['decimals'], cfg4['chosen_ratio']
PERTURB = ['net_connect', 'file_create', 'reg_write', 'reg_create_key',
           'reg_delete_key', 'reg_delete_value', 'duration_s']
EVENT9 = ['process_create', 'process_term', 'image_load', 'net_connect', 'file_create',
          'reg_create_key', 'reg_write', 'reg_delete_key', 'reg_delete_value']

df = pd.read_pickle(P3 + 'df.pkl')
FEAT = json.load(open(P3 + 'featureset.json'))['features_final']
tr = np.load(P3 + 'tr_idx.npy')
y = (df['label'] == 'malicious').astype(int).values
Xall = df[FEAT].values; COL = {c: i for i, c in enumerate(FEAT)}
mem_aud = np.load(P4 + 'mia_audit_mem.npy'); non_aud = np.load(P4 + 'mia_audit_non.npy')
print(f'고정 감사셋: member {len(mem_aud)} / non-member {len(non_aud)}\n', flush=True)

def perturb(Xs, f, lo, hi, s=0.20):
    j = COL[f]; Xp = Xs.copy(); Xp[:, j] = Xp[:, j] * (1 - s)
    v = (Xp[:, j] >= 0) & (Xp[:, j] >= lo[j]) & (Xp[:, j] <= hi[j])
    if f in EVENT9:
        Xp[:, COL['n_events']] = Xs[:, COL['n_events']] - (Xs[:, j] - Xp[:, j]); v &= Xp[:, COL['n_events']] >= 0
    return Xp, v

def train(kind, idx, seed):
    """kind별 학습 절차. 반환: 확률 함수"""
    X, yy = Xall[idx], y[idx]
    if kind == 'A_none':
        m = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1).fit(X, yy)
        return lambda Z: m.predict_proba(Z)[:, 1]
    if kind == 'Selective':
        m = RandomForestClassifier(n_estimators=N_EST, random_state=seed, n_jobs=-1).fit(X, yy)
        return lambda Z: m.predict_proba(Z)[:, 1]
    # Batch: 정규화 + 이산화 + 교란학습 (학습 데이터 자신에서 flip 풀 생성)
    lo, hi = X.min(0), X.max(0)
    m0 = RandomForestClassifier(n_estimators=N_EST, min_samples_leaf=MSL, random_state=seed, n_jobs=-1).fit(X, yy)
    pr = lambda mm, Z: np.round(mm.predict_proba(Z)[:, 1], DEC)
    tb = (yy == 1) & (pr(m0, X) >= 0.5); pool = []
    for f in PERTURB:
        Xp, v = perturb(X[tb], f, lo, hi); pool.append(Xp[v & (pr(m0, Xp) < 0.5)])
    pool = np.vstack(pool)
    n_aug = int(round(int((yy == 1).sum()) * RATIO))
    aug = pool[np.random.RandomState(seed).choice(len(pool), n_aug, replace=n_aug > len(pool))] if len(pool) else pool
    m = RandomForestClassifier(n_estimators=N_EST, min_samples_leaf=MSL, random_state=seed, n_jobs=-1).fit(
        np.vstack([X, aug]), np.concatenate([yy, np.ones(len(aug), dtype=int)]))
    return lambda Z: pr(m, Z)

def feats(p1, yt):
    proba = np.column_stack([1 - p1, p1])
    conf = proba.max(1); pt = proba[np.arange(len(yt)), yt]
    ent = -(proba * np.log(proba + 1e-12)).sum(1)
    return np.column_stack([conf, pt, ent])

SC = {}
for kind in ['A_none', 'Selective', 'Batch']:
    rs = np.random.RandomState(SEED); Xs, ys = [], []
    for k in range(5):                                   # shadow 5개 (§5.3)
        perm = rs.permutation(len(tr)); i_in, i_out = tr[perm[:12000]], tr[perm[12000:]]
        f = train(kind, i_in, seed=1000 + k)
        for idx, lab in [(i_in, 1), (i_out, 0)]:
            Xs.append(feats(f(Xall[idx]), y[idx])); ys.append(np.full(len(idx), lab))
    atk = LogisticRegression(max_iter=1000).fit(np.vstack(Xs), np.concatenate(ys))
    tgt = train(kind, tr, seed=SEED)
    sc = atk.predict_proba(np.vstack([feats(tgt(Xall[mem_aud]), y[mem_aud]),
                                      feats(tgt(Xall[non_aud]), y[non_aud])]))[:, 1]
    SC[kind] = sc
    a = roc_auc_score(np.r_[np.ones(len(mem_aud)), np.zeros(len(non_aud))], sc)
    print(f'  {kind:10s} MIA AUC_adv = {max(a, 1-a):.4f}', flush=True)

ya = np.r_[np.ones(len(mem_aud)), np.zeros(len(non_aud))]
def aadv(s, i):
    a = roc_auc_score(ya[i], s[i]); return max(a, 1 - a)
pt = {k: aadv(SC[k], np.arange(len(ya))) for k in SC}

# 대응 부트스트랩 (감사셋 표본 재표집, 클래스 층화)
rng = np.random.RandomState(SEED)
im, inn = np.where(ya == 1)[0], np.where(ya == 0)[0]
d = []
for _ in range(N_BOOT):
    i = np.r_[rng.choice(im, len(im)), rng.choice(inn, len(inn))]
    d.append(aadv(SC['Selective'], i) - aadv(SC['Batch'], i))
ci = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
res = dict(point=pt, diff_sel_minus_batch=dict(est=pt['Selective'] - pt['Batch'], ci=ci))
json.dump(res, open(OUT + 'h5_eval_part2_mia.json', 'w'), indent=2)
print(f"\n  AUC_adv 차이(Selective − Batch) Δ={res['diff_sel_minus_batch']['est']:+.4f}  CI [{ci[0]:+.4f}, {ci[1]:+.4f}]")
