"""Phase 6c - 공식 Test 시간적 일반화 보고 (코호트별 분리, §11.2.1 / §8 / §9)
- 봉인 해제 전 해시 검증
- H5 판정에 사용하지 않음. 시간적 일반화 서술 전용
- 코호트는 단일 클래스이므로 MCC 대신: 악성 코호트=탐지율(Recall), 정상 코호트=특이도/FPR
- 임계값 0.5 고정 (봉인 셋에서 재조정 금지, §9)
"""
import pandas as pd, numpy as np, json, hashlib, sys, warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import matthews_corrcoef

P3 = '/home/claude/phase3/'; OUT = '/home/claude/phase6/'
SEED = 42; EXPECTED = 'c63d8ca03e7a4480'

off = np.load(P3 + 'offtest_idx.npy')
got = hashlib.sha256(np.sort(np.asarray(off)).tobytes()).hexdigest()[:16]
print(f'공식 Test 해시: 기대 {EXPECTED} / 실제 {got} -> {"일치 ✅ 봉인 해제" if got == EXPECTED else "불일치 ❌"}', flush=True)
if got != EXPECTED: sys.exit('해시 불일치 — 중단')

cfg23 = json.load(open(OUT + 'step23.json')); cfg4 = json.load(open(OUT + 'step4_perturb.json'))
N_EST, MSL, DEC, RATIO = cfg23['n_estimators'], cfg23['min_samples_leaf'], cfg23['decimals'], cfg4['chosen_ratio']
BP = json.load(open(P3 + 'bo_result.json'))['best_params']
PERTURB = ['net_connect', 'file_create', 'reg_write', 'reg_create_key',
           'reg_delete_key', 'reg_delete_value', 'duration_s']
EVENT9 = ['process_create', 'process_term', 'image_load', 'net_connect', 'file_create',
          'reg_create_key', 'reg_write', 'reg_delete_key', 'reg_delete_value']

df = pd.read_pickle(P3 + 'df.pkl')
FEAT = json.load(open(P3 + 'featureset.json'))['features_final']
tr = np.load(P3 + 'tr_idx.npy')
y = (df['label'] == 'malicious').astype(int).values
Xtr, ytr = df.loc[tr, FEAT].values, y[tr]
Xo, yo = df.loc[off, FEAT].values, y[off]
yr = df.loc[off, 'year'].fillna(-1).astype(int).values
COL = {c: i for i, c in enumerate(FEAT)}; lo, hi = Xtr.min(0), Xtr.max(0)
sc = StandardScaler().fit(Xtr)

# ---------- 모델 6종 (Phase 3: A/B/C/D, Phase 6: Selective/Batch) ----------
M = {}
M['A_RF'] = lambda X, m=RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1).fit(Xtr, ytr): m.predict_proba(X)[:, 1]
M['B_RF_BO'] = lambda X, m=RandomForestClassifier(**BP, random_state=SEED, n_jobs=-1).fit(Xtr, ytr): m.predict_proba(X)[:, 1]
_d = MLPClassifier(max_iter=200, early_stopping=True, n_iter_no_change=10, random_state=SEED).fit(sc.transform(Xtr), ytr)
M['D_MLP'] = lambda X: _d.predict_proba(sc.transform(X))[:, 1]
bases = [RandomForestClassifier(**BP, random_state=SEED + k, n_jobs=-1) for k in range(3)]
bases.append(HistGradientBoostingClassifier(random_state=SEED))
oof = np.column_stack([cross_val_predict(b, Xtr, ytr, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                                         method='predict_proba', n_jobs=1)[:, 1] for b in bases])
meta = LogisticRegression(max_iter=1000).fit(oof, ytr)
for b in bases: b.fit(Xtr, ytr)
M['C_STACK'] = lambda X: meta.predict_proba(np.column_stack([b.predict_proba(X)[:, 1] for b in bases]))[:, 1]
_s = RandomForestClassifier(n_estimators=N_EST, random_state=SEED, n_jobs=-1).fit(Xtr, ytr)
M['Selective'] = lambda X: _s.predict_proba(X)[:, 1]

def perturb(Xs, f, s=0.20):
    j = COL[f]; Xp = Xs.copy(); Xp[:, j] = Xp[:, j] * (1 - s)
    v = (Xp[:, j] >= 0) & (Xp[:, j] >= lo[j]) & (Xp[:, j] <= hi[j])
    if f in EVENT9:
        Xp[:, COL['n_events']] = Xs[:, COL['n_events']] - (Xs[:, j] - Xp[:, j]); v &= Xp[:, COL['n_events']] >= 0
    return Xp, v
m0 = RandomForestClassifier(n_estimators=N_EST, min_samples_leaf=MSL, random_state=SEED, n_jobs=-1).fit(Xtr, ytr)
pr0 = lambda Z: np.round(m0.predict_proba(Z)[:, 1], DEC)
tb = (ytr == 1) & (pr0(Xtr) >= 0.5); pool = []
for f in PERTURB:
    Xp, v = perturb(Xtr[tb], f); pool.append(Xp[v & (pr0(Xp) < 0.5)])
pool = np.vstack(pool); rs = np.random.RandomState(SEED)
for r in [0.10, 0.20, 0.30, 0.40, 0.50]:
    n_aug = int(round(int((ytr == 1).sum()) * r))
    aug = pool[rs.choice(len(pool), n_aug, replace=n_aug > len(pool))]
    if r == RATIO: AUG = aug
_b = RandomForestClassifier(n_estimators=N_EST, min_samples_leaf=MSL, random_state=SEED, n_jobs=-1).fit(
    np.vstack([Xtr, AUG]), np.concatenate([ytr, np.ones(len(AUG), dtype=int)]))
M['Batch'] = lambda X: np.round(_b.predict_proba(X)[:, 1], DEC)
print('모델 6종 준비 완료\n', flush=True)

# ---------- 코호트 정의 ----------
COH = {'2017_악성': (yr == 2017) & (yo == 1),
       '2024_악성': (yr == 2024) & (yo == 1),
       '2024_정상': (yr == 2024) & (yo == 0)}
other = ~(COH['2017_악성'] | COH['2024_악성'] | COH['2024_정상'])
print('코호트 규모:', {k: int(v.sum()) for k, v in COH.items()}, '| 기타:', int(other.sum()))
print(f'통합 악성 비율: {yo.mean()*100:.1f}%  (Train/Val은 50.0%, §11.2.2 사전확률 이동)\n')

def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n; d = 1 + z*z/n; c = (p + z*z/(2*n)) / d; h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (c - h, c + h)

res = {}
print(f"{'모델':10s} {'통합MCC':>8s} | {'2017악성 탐지율':>18s} | {'2024악성 탐지율':>18s} | {'2024정상 FPR':>18s}")
for k, f in M.items():
    p = f(Xo); pred = (p >= 0.5).astype(int)
    row = dict(MCC_all=float(matthews_corrcoef(yo, pred)))
    for c, mask in COH.items():
        yc, pc = yo[mask], pred[mask]
        if c.endswith('악성'):
            kk = int((pc == 1).sum()); nn = len(pc); rate = kk / nn
            row[c] = dict(metric='recall', value=rate, ci=wilson(kk, nn), n=nn)
        else:
            kk = int((pc == 1).sum()); nn = len(pc); rate = kk / nn
            row[c] = dict(metric='FPR', value=rate, ci=wilson(kk, nn), n=nn)
    res[k] = row
    s = lambda c: f"{row[c]['value']:.4f}[{row[c]['ci'][0]:.3f},{row[c]['ci'][1]:.3f}]"
    print(f"{k:10s} {row['MCC_all']:8.4f} | {s('2017_악성'):>18s} | {s('2024_악성'):>18s} | {s('2024_정상'):>18s}", flush=True)

# 시간적 일반화 격차: 2017 악성 탐지율 - 2024 악성 탐지율
print('\n=== 시간적 일반화 격차 (2017 악성 탐지율 − 2024 악성 탐지율) ===')
for k in M:
    g = res[k]['2017_악성']['value'] - res[k]['2024_악성']['value']
    res[k]['temporal_gap'] = float(g)
    print(f'  {k:10s} {g:+.4f}')

# ---------- 진단: 2024 코호트 내부 ROC-AUC (임계값 비의존 구분 능력) ----------
from sklearn.metrics import roc_auc_score
m24 = (yr == 2024)
print('\n=== 2024 코호트 내부 ROC-AUC (악성 vs 정상, 연도 교란 없음, 임계값 비의존) ===')
for k, f in M.items():
    p = f(Xo)
    auc = roc_auc_score(yo[m24], p[m24])
    mp_mal = p[m24 & (yo == 1)].mean(); mp_ben = p[m24 & (yo == 0)].mean()
    res[k]['auc_2024'] = float(auc)
    res[k]['mean_prob_2024_mal'] = float(mp_mal); res[k]['mean_prob_2024_ben'] = float(mp_ben)
    print(f'  {k:10s} AUC={auc:.4f}   평균 악성확률: 2024악성 {mp_mal:.4f} / 2024정상 {mp_ben:.4f}')
json.dump(res, open(OUT + 'official_test_cohort.json', 'w'), indent=2, default=float)
