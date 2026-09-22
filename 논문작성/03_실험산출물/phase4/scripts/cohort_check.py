"""교란 영향 크기 측정 — 연도 범위가 좁고 클래스가 균형인 구간에서의 탐지 성능
탐색적 부가 분석. Train+Validation만 사용 (봉인 셋 미사용).

비교 설계
  (a) 전체 구간 (2007~2024) — Phase 3와 동일 조건
  (b) 2020-2022 코호트 — 연도 폭 3년, 악성 53.3%
  (c) 2020-2022 + 연도별 클래스 균형 강제 — 코호트 내부 잔여 교란까지 제거

표본 수 차이가 성능에 미치는 영향을 분리하기 위해, (a)는 (b)와 동일 크기로 하위표집한 조건도 함께 측정한다.
"""
import pandas as pd, numpy as np, json, warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import matthews_corrcoef, roc_auc_score

P3 = '/home/claude/phase3/'
SEED = 42
rng = np.random.RandomState(SEED)

df = pd.read_pickle(P3 + 'df.pkl')
FEAT = json.load(open(P3 + 'featureset.json'))['features_final']
BP = json.load(open(P3 + 'bo_result.json'))['best_params']
tr = np.load(P3 + 'tr_idx.npy'); va = np.load(P3 + 'va_idx.npy')
dev = np.concatenate([tr, va])

d = df.loc[dev].copy()
d = d[d.year.notna()].reset_index(drop=True)
d['yr'] = d.year.astype(int)
d['lab'] = (d.label == 'malicious').astype(int)

cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

def run(sub, tag):
    """동일 모델(B의 BO 최적 파라미터)로 5-fold 교차검증 성능 산출"""
    X = sub[FEAT].values; y = sub['lab'].values
    if len(np.unique(y)) < 2:
        print(f'{tag}: 단일 클래스 — 건너뜀'); return None
    m = RandomForestClassifier(**BP, random_state=SEED, n_jobs=-1)
    prob = cross_val_predict(m, X, y, cv=cv, method='predict_proba', n_jobs=1)[:, 1]
    mcc = matthews_corrcoef(y, (prob >= 0.5).astype(int))
    auc = roc_auc_score(y, prob)
    # 같은 데이터에서 "연도만으로 라벨 예측" AUC (교란 크기 지표)
    yr_auc = roc_auc_score(y, -sub['yr'].values) if sub['yr'].nunique() > 1 else float('nan')
    print(f'{tag:34s} n={len(y):6d} 악성 {y.mean()*100:5.1f}%  '
          f'MCC={mcc:.4f}  AUC={auc:.4f}  연도단독AUC={yr_auc:.4f}')
    return dict(tag=tag, n=int(len(y)), mal_ratio=float(y.mean()),
                MCC=float(mcc), AUC=float(auc), year_only_AUC=float(yr_auc))

res = []

# (a) 전체 구간
res.append(run(d, '(a) 전체 2007-2024'))

# (b) 2020-2022 코호트
c = d[d.yr.between(2020, 2022)]
res.append(run(c, '(b) 2020-2022 코호트'))

# (a') 전체 구간을 (b)와 동일 크기로 하위표집 (표본 수 효과 분리)
idx = rng.choice(len(d), len(c), replace=False)
res.append(run(d.iloc[idx], "(a') 전체, (b)와 동일 크기"))

# (c) 2020-2022 + 연도별 클래스 균형 강제
keep = []
for yv in sorted(c.yr.unique()):
    s = c[c.yr == yv]
    n = min((s.lab == 1).sum(), (s.lab == 0).sum())
    if n == 0: continue
    keep.append(s[s.lab == 1].sample(n, random_state=SEED))
    keep.append(s[s.lab == 0].sample(n, random_state=SEED))
cb = pd.concat(keep)
res.append(run(cb, '(c) 2020-2022 + 연도별 균형'))

# (d) 단일 연도 (교란 완전 제거) — 가장 표본이 많은 연도
best_y = c.groupby('yr').size().idxmax()
s1 = c[c.yr == best_y]
res.append(run(s1, f'(d) {best_y}년 단독'))

json.dump([r for r in res if r], open('/home/claude/phase4/cohort_check.json', 'w'), indent=2)

print()
print('해석 기준: (b)~(d)의 MCC가 (a)와 비슷하면 모델이 실제 행위를 학습한다는 증거.')
print('           크게 낮아지면 (a)의 성능이 연도 교란에 상당 부분 기인.')
