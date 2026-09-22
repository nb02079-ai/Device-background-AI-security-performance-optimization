"""Phase 6a-3 - 일괄 개선 단계 4: 교란학습 (Validation 선택)
§10.3A ③: Train 악성 샘플에서 새로 생성(누수 수정), 20% 감소, flip 샘플만 채택,
증강 규모 = Train 악성(12,000)의 {10,20,30,40,50}%, 부족 시 복원 추출
선택 기준: Validation 최대 flip rate(20%, 7특징 중 최대) 작은 순 -> MCC 큰 순
"""
import pandas as pd, numpy as np, json, warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef

P3 = '/home/claude/phase3/'; OUT = '/home/claude/phase6/'
SEED = 42
cfg = json.load(open(OUT + 'step23.json'))
N_EST, MSL, DEC = cfg['n_estimators'], cfg['min_samples_leaf'], cfg['decimals']
PERTURB = ['net_connect', 'file_create', 'reg_write', 'reg_create_key',
           'reg_delete_key', 'reg_delete_value', 'duration_s']
EVENT9 = ['process_create', 'process_term', 'image_load', 'net_connect', 'file_create',
          'reg_create_key', 'reg_write', 'reg_delete_key', 'reg_delete_value']
RATIOS = [0.10, 0.20, 0.30, 0.40, 0.50]

df = pd.read_pickle(P3 + 'df.pkl')
FEAT = json.load(open(P3 + 'featureset.json'))['features_final']
tr = np.load(P3 + 'tr_idx.npy'); va = np.load(P3 + 'va_idx.npy')
y = (df['label'] == 'malicious').astype(int).values
Xtr, ytr = df.loc[tr, FEAT].values, y[tr]; Xva, yva = df.loc[va, FEAT].values, y[va]
COL = {c: i for i, c in enumerate(FEAT)}
tr_min, tr_max = Xtr.min(0), Xtr.max(0)
N_MAL = int((ytr == 1).sum())
print(f'설정: n_estimators={N_EST}, min_samples_leaf={MSL}, 이산화 {DEC}자리 / Train 악성 {N_MAL}\n')

def rf(X, yy): return RandomForestClassifier(n_estimators=N_EST, min_samples_leaf=MSL,
                                              random_state=SEED, n_jobs=-1).fit(X, yy)
def prob(m, X): return np.round(m.predict_proba(X)[:, 1], DEC)

def perturb(Xs, f, s=0.20):
    """§6.2 유효성 검사 포함 교란. (교란본, 유효마스크) 반환"""
    j = COL[f]; Xp = Xs.copy(); Xp[:, j] = Xp[:, j] * (1 - s)
    valid = (Xp[:, j] >= 0) & (Xp[:, j] >= tr_min[j]) & (Xp[:, j] <= tr_max[j])
    if f in EVENT9:
        Xp[:, COL['n_events']] = Xs[:, COL['n_events']] - (Xs[:, j] - Xp[:, j])
        valid &= Xp[:, COL['n_events']] >= 0
    return Xp, valid

def max_flip(m, X, yy):
    """정확히 악성 분류된 표본 대상, 7특징 중 최대 flip rate (§10.3A ⑥)"""
    p0 = prob(m, X); base = (yy == 1) & (p0 >= 0.5); Xb = X[base]
    rates = {}
    for f in PERTURB:
        Xp, v = perturb(Xb, f)
        rates[f] = float((prob(m, Xp[v]) < 0.5).mean()) if v.sum() else 0.0
    return max(rates.values()), rates

# ---- 기준: 교란학습 없는 모델 (단계 3까지) ----
m0 = rf(Xtr, ytr)
f0, r0 = max_flip(m0, Xva, yva)
mcc0 = matthews_corrcoef(yva, (prob(m0, Xva) >= 0.5).astype(int))
print(f'교란학습 전: Val 최대 flip={f0:.4f}  MCC={mcc0:.4f}', flush=True)

# ---- flip 샘플 풀 생성: Train 악성에서 (Validation 미사용) ----
ptr = prob(m0, Xtr); tb = (ytr == 1) & (ptr >= 0.5); Xtb = Xtr[tb]
pool = []
for f in PERTURB:
    Xp, v = perturb(Xtb, f)
    flipped = v & (prob(m0, Xp) < 0.5)
    pool.append(Xp[flipped])
pool = np.vstack(pool)
print(f'flip 샘플 풀 (Train 기원): {len(pool)}건\n', flush=True)

rs = np.random.RandomState(SEED)
rows = []
for r in RATIOS:
    n_aug = int(round(N_MAL * r))
    replace = n_aug > len(pool)
    aug = pool[rs.choice(len(pool), n_aug, replace=replace)]
    m = rf(np.vstack([Xtr, aug]), np.concatenate([ytr, np.ones(n_aug, dtype=int)]))
    fr, _ = max_flip(m, Xva, yva)
    mcc = matthews_corrcoef(yva, (prob(m, Xva) >= 0.5).astype(int))
    rows.append(dict(ratio=r, n_aug=n_aug, with_replacement=bool(replace),
                     replacement_factor=round(n_aug / len(pool), 2),
                     val_max_flip=fr, val_MCC=float(mcc)))
    print(f'  증강 {int(r*100):2d}% ({n_aug:5d}건, 복원배율 {n_aug/len(pool):5.1f}x)  '
          f'최대flip={fr:.4f}  MCC={mcc:.4f}', flush=True)

chosen = sorted(rows, key=lambda x: (x['val_max_flip'], -x['val_MCC']))[0]
print(f'\n  -> 선택 증강 {int(chosen["ratio"]*100)}%', flush=True)

json.dump({'before': dict(val_max_flip=f0, val_MCC=float(mcc0), per_feature=r0),
           'pool_size': int(len(pool)), 'candidates': rows, 'chosen_ratio': chosen['ratio']},
          open(OUT + 'step4_perturb.json', 'w'), indent=2)
