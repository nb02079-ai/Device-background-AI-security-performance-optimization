"""간접 교란 확인 — 행위 특징이 연도 정보를 얼마나 담고 있는가
프로토콜 §11.2.3 한계 검증용. Train+Validation만 사용 (봉인 셋 미사용).

세 가지를 측정한다.
 (1) 행위 특징 12개로 연도를 예측할 수 있는가 (회귀 R^2, 분류 AUC)
 (2) 클래스 내부에서도 연도가 예측되는가 (라벨 경로가 아닌 직접 경로 확인)
 (3) 특징별 연도 상관
"""
import pandas as pd, numpy as np, json, warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, r2_score
from scipy.stats import spearmanr

P3 = '/home/claude/phase3/'
df = pd.read_pickle(P3 + 'df.pkl')
FEAT = json.load(open(P3 + 'featureset.json'))['features_final']
tr = np.load(P3 + 'tr_idx.npy'); va = np.load(P3 + 'va_idx.npy')
dev = np.concatenate([tr, va])

d = df.loc[dev].copy()
d = d[d.year.notna()]
X = d[FEAT].values
yr = d.year.astype(int).values
lab = (d.label == 'malicious').astype(int).values

print(f'대상: Train+Validation {len(d)}건 (봉인 셋 미사용)\n')

# ---------- (1) 행위 특징 -> 연도 회귀 ----------
print('=== (1) 행위 특징 12개로 연도 예측 ===')
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
pred = cross_val_predict(rf, X, yr, cv=KFold(5, shuffle=True, random_state=42), n_jobs=1)
print(f'  연도 회귀 R^2      : {r2_score(yr, pred):.4f}')
print(f'  평균절대오차(년)   : {np.abs(pred - yr).mean():.2f}')
print(f'  기준선(연도 평균)  : MAE {np.abs(yr.mean() - yr).mean():.2f}')

# 2017 vs 그외 이진 분류
is17 = (yr == 2017).astype(int)
auc17 = cross_val_score(RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
                        X, is17, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                        scoring='roc_auc', n_jobs=1).mean()
print(f'  "2017년인가" 분류 AUC: {auc17:.4f}')

# ---------- (2) 클래스 내부 (라벨 경로 차단) ----------
print('\n=== (2) 클래스 내부에서의 연도 예측 (간접 경로 확인) ===')
print('  라벨을 고정하면 라벨을 통한 연도 예측 경로가 차단된다.')
for name, m in [('악성만', lab == 1), ('정상만', lab == 0)]:
    Xs, ys = X[m], yr[m]
    if len(np.unique(ys)) < 2:
        print(f'  {name}: 연도가 단일값'); continue
    is17s = (ys == 2017).astype(int)
    if is17s.sum() < 50 or (1 - is17s).sum() < 50:
        print(f'  {name}: 2017 {is17s.sum()} / 그외 {(1-is17s).sum()} — 표본 부족, 회귀만 수행')
        a = float('nan')
    else:
        a = cross_val_score(RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
                            Xs, is17s, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                            scoring='roc_auc', n_jobs=1).mean()
    p = cross_val_predict(RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
                          Xs, ys, cv=KFold(5, shuffle=True, random_state=42), n_jobs=1)
    print(f'  {name} (n={len(ys)}): 연도 회귀 R^2={r2_score(ys,p):.4f}  "2017?" AUC={a:.4f}')

# ---------- (3) 특징별 연도 상관 ----------
print('\n=== (3) 특징별 연도 Spearman 상관 (|rho| 상위) ===')
rows = []
for i, c in enumerate(FEAT):
    rho, _ = spearmanr(X[:, i], yr)
    rho_m, _ = spearmanr(X[lab == 1, i], yr[lab == 1])
    rows.append((c, rho, rho_m))
for c, r, rm in sorted(rows, key=lambda x: -abs(x[1]))[:6]:
    print(f'  {c:20s} 전체 {r:+.3f}   악성내부 {rm:+.3f}')

json.dump({'year_r2': float(r2_score(yr, pred)), 'is2017_auc': float(auc17),
           'feature_year_rho': {c: float(r) for c, r, _ in rows}},
          open('/home/claude/phase4/year_confound.json', 'w'), indent=2)
