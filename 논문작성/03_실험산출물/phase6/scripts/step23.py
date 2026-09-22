"""Phase 6a-2 - 일괄 개선 경로 단계 2~4 선택 (Validation)
단계 2: 정규화   min_samples_leaf {2,5,10} -> 확신도 격차로 선택 (§10.3A ④)
단계 3: 이산화   소수점 {1,2}자리          -> MIA 선택셋으로 선택 (§10.3A ④)
단계 4: 교란학습 증강비율 {10..50}%        -> Validation flip rate로 선택 (§10.3A ③)
일괄 개선은 허용 한도 필터 미적용: 각 단계를 반드시 적용하되 설정만 최적화 (§10.3A ②-1)
H5-Test·공식 Test·MIA 최종감사셋 미사용.
"""
import pandas as pd, numpy as np, json, os, sys, warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, roc_auc_score, confusion_matrix

P3 = '/home/claude/phase3/'; P4 = '/home/claude/phase4/'; OUT = '/home/claude/phase6/'
SEED = 42
N_EST = json.load(open(OUT + 'step1_lightweight.json'))['chosen']['n_estimators']
PERTURB = ['net_connect', 'file_create', 'reg_write', 'reg_create_key',
           'reg_delete_key', 'reg_delete_value', 'duration_s']
EVENT9 = ['process_create', 'process_term', 'image_load', 'net_connect', 'file_create',
          'reg_create_key', 'reg_write', 'reg_delete_key', 'reg_delete_value']

df = pd.read_pickle(P3 + 'df.pkl')
FEAT = json.load(open(P3 + 'featureset.json'))['features_final']
tr = np.load(P3 + 'tr_idx.npy'); va = np.load(P3 + 'va_idx.npy')
y = (df['label'] == 'malicious').astype(int).values
Xall = df[FEAT].values
year = df['year'].fillna(-1).astype(int).values
Xtr, ytr = Xall[tr], y[tr]; Xva, yva = Xall[va], y[va]
COL = {c: i for i, c in enumerate(FEAT)}
tr_min, tr_max = Xtr.min(0), Xtr.max(0)

# ---------- MIA 분할 재구성 (mia.py와 동일한 난수 호출 순서) ----------
rng = np.random.RandomState(SEED)
def match_year(ia, ib, rs):
    ya, yb = year[ia], year[ib]; ka, kb = [], []
    for v in np.unique(np.concatenate([ya, yb])):
        aa, bb = ia[ya == v], ib[yb == v]; n = min(len(aa), len(bb))
        if n == 0: continue
        ka.append(rs.choice(aa, n, replace=False)); kb.append(rs.choice(bb, n, replace=False))
    return np.concatenate(ka), np.concatenate(kb)
def split3(idx, rs):
    p = rs.permutation(len(idx)); idx = idx[p]; n = len(idx)
    return idx[:int(n*.5)], idx[int(n*.5):int(n*.75)], idx[int(n*.75):]
mem_all, non_all = match_year(tr, va, rng)
mem_atk, mem_sel, mem_aud = split3(mem_all, rng)
non_atk, non_sel, non_aud = split3(non_all, rng)

# 무결성 검증: 재구성한 감사셋이 Phase 4에서 저장한 것과 일치해야 한다
ok = (np.array_equal(np.sort(mem_aud), np.sort(np.load(P4 + 'mia_audit_mem.npy'))) and
      np.array_equal(np.sort(non_aud), np.sort(np.load(P4 + 'mia_audit_non.npy'))))
print(f'MIA 분할 재구성 무결성: {"일치 ✅" if ok else "불일치 ❌"}', flush=True)
if not ok: sys.exit('재구성 실패 — 중단')
print(f'  선택셋 member {len(mem_sel)} / non-member {len(non_sel)}  (감사셋은 사용하지 않음)\n')

def rf(msl, X=Xtr, yy=ytr, seed=SEED):
    return RandomForestClassifier(n_estimators=N_EST, min_samples_leaf=msl,
                                  random_state=seed, n_jobs=-1).fit(X, yy)

def disc(p, d):
    return p if d is None else np.round(p, d)

# ================= 단계 2: 정규화 (확신도 격차) =================
print('=== 단계 2: 정규화 min_samples_leaf (확신도 격차로 선택) ===', flush=True)
s2 = []
for msl in [2, 5, 10]:
    m = rf(msl)
    gap = m.predict_proba(Xall[mem_sel]).max(1).mean() - m.predict_proba(Xall[non_sel]).max(1).mean()
    mcc = matthews_corrcoef(yva, m.predict(Xva))
    s2.append(dict(msl=msl, conf_gap=float(gap), val_MCC=float(mcc)))
    print(f'  msl={msl:2d}  확신도격차={gap:+.4f}  Val MCC={mcc:.4f}', flush=True)
# 타이브레이크: 목표축(격차) 개선폭 큰 순 = 격차 작은 순 -> 부작용(MCC) 적은 순
MSL = sorted(s2, key=lambda r: (r['conf_gap'], -r['val_MCC']))[0]['msl']
print(f'  -> 선택 min_samples_leaf={MSL}\n', flush=True)

# ================= 단계 3: 이산화 (MIA 선택셋) =================
print('=== 단계 3: 출력 이산화 (MIA 선택셋으로 선택) ===', flush=True)
def attack_feat(proba, yt):
    conf = proba.max(1); pt = proba[np.arange(len(yt)), yt]
    ent = -(proba * np.log(proba + 1e-12)).sum(1)
    return np.column_stack([conf, pt, ent])

def mia_auc_on_selection(d):
    """shadow 5개로 공격모델 학습 후 선택셋에서 AUC_adv 산출"""
    rs = np.random.RandomState(SEED)
    Xs, ys = [], []
    for k in range(5):
        perm = rs.permutation(len(tr)); i_in, i_out = tr[perm[:12000]], tr[perm[12000:]]
        sm = rf(MSL, Xall[i_in], y[i_in], seed=1000 + k)
        for idx, lab in [(i_in, 1), (i_out, 0)]:
            pr = sm.predict_proba(Xall[idx]); pr = np.column_stack([1 - disc(pr[:, 1], d), disc(pr[:, 1], d)])
            Xs.append(attack_feat(pr, y[idx])); ys.append(np.full(len(idx), lab))
    atk = LogisticRegression(max_iter=1000).fit(np.vstack(Xs), np.concatenate(ys))
    tgt = rf(MSL)
    def f(idx):
        pr = tgt.predict_proba(Xall[idx]); pr = np.column_stack([1 - disc(pr[:, 1], d), disc(pr[:, 1], d)])
        return attack_feat(pr, y[idx])
    Xa = np.vstack([f(mem_sel), f(non_sel)])
    ya = np.concatenate([np.ones(len(mem_sel)), np.zeros(len(non_sel))])
    a = roc_auc_score(ya, atk.predict_proba(Xa)[:, 1])
    return float(max(a, 1 - a))

s3 = []
for d in [1, 2]:
    auc = mia_auc_on_selection(d)
    m = rf(MSL); mcc = matthews_corrcoef(yva, (disc(m.predict_proba(Xva)[:, 1], d) >= 0.5).astype(int))
    s3.append(dict(decimals=d, mia_auc_adv_sel=auc, val_MCC=float(mcc)))
    print(f'  소수점 {d}자리  MIA AUC_adv(선택셋)={auc:.4f}  Val MCC={mcc:.4f}', flush=True)
DEC = sorted(s3, key=lambda r: (r['mia_auc_adv_sel'], -r['val_MCC']))[0]['decimals']
print(f'  -> 선택 소수점 {DEC}자리\n', flush=True)

json.dump({'n_estimators': N_EST, 'step2': s2, 'min_samples_leaf': MSL,
           'step3': s3, 'decimals': DEC, 'mia_split_integrity': ok},
          open(OUT + 'step23.json', 'w'), indent=2)
print('단계 2~3 저장 완료')
