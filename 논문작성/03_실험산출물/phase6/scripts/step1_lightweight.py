"""Phase 6a-1 - 경량화 단계 선택 (Validation). 선택적·일괄 두 경로가 공유.
프로토콜 v3.4 §10.3A ⑦: n_estimators 10개 후보
선택 규칙: §10.2 타이브레이킹(목표축 개선폭 큰 순 -> 부작용 적은 순) + 허용 한도
허용 한도: MCC 절대 저하 <= 0.05, FPR 상승 <= 10%p (지연 증가는 경량화라 해당 없음)
H5-Test·공식 Test 미사용.
"""
import pandas as pd, numpy as np, json, time, os, warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef, confusion_matrix

P3 = '/home/claude/phase3/'; OUT = '/home/claude/phase6/'
os.makedirs(OUT, exist_ok=True)
SEEDS = [42, 7, 123, 2024, 777]
CANDS = [20, 29, 38, 47, 56, 64, 73, 82, 91, 100]

df = pd.read_pickle(P3 + 'df.pkl')
FEAT = json.load(open(P3 + 'featureset.json'))['features_final']
tr = np.load(P3 + 'tr_idx.npy'); va = np.load(P3 + 'va_idx.npy')
y = (df['label'] == 'malicious').astype(int).values
Xtr, ytr = df.loc[tr, FEAT].values, y[tr]
Xva, yva = df.loc[va, FEAT].values, y[va]

def evaluate(m):
    p = m.predict_proba(Xva)[:, 1]; pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(yva, pred).ravel()
    Xb = Xva[:10]
    for _ in range(5): m.predict_proba(Xb)            # 워밍업
    ts = []
    for _ in range(200):
        t0 = time.perf_counter(); m.predict_proba(Xb); ts.append(time.perf_counter() - t0)
    return dict(MCC=matthews_corrcoef(yva, pred), FPR=fp / (fp + tn),
                lat=float(np.median(ts) * 1000 / 10))   # batch=10, ms/샘플, p50

# ---- 기준선: 모델 A (n_estimators=300) ----
base = [evaluate(RandomForestClassifier(n_estimators=300, random_state=s, n_jobs=1).fit(Xtr, ytr))
        for s in SEEDS]
B_MCC = np.mean([b['MCC'] for b in base]); B_FPR = np.mean([b['FPR'] for b in base])
B_LAT = np.mean([b['lat'] for b in base])
print(f'기준선 A: MCC={B_MCC:.4f} FPR={B_FPR:.4f} 지연={B_LAT:.4f}ms\n', flush=True)

rows = []
for k in CANDS:
    rs = [evaluate(RandomForestClassifier(n_estimators=k, random_state=s, n_jobs=1).fit(Xtr, ytr))
          for s in SEEDS]
    mcc = np.mean([r['MCC'] for r in rs]); fpr = np.mean([r['FPR'] for r in rs])
    lat = np.mean([r['lat'] for r in rs])
    red = (B_LAT - lat) / B_LAT                      # 지연 상대 감소율
    ok_mcc = (B_MCC - mcc) <= 0.05
    ok_fpr = (fpr - B_FPR) <= 0.10
    success = red >= 0.20                             # §8 리소스 성공 기준
    rows.append(dict(n_estimators=k, MCC=float(mcc), FPR=float(fpr), lat_ms=float(lat),
                     lat_reduction=float(red), dMCC=float(mcc - B_MCC),
                     pass_allowance=bool(ok_mcc and ok_fpr), success=bool(success)))
    print(f'n_est={k:3d}  MCC={mcc:.4f}(Δ{mcc-B_MCC:+.4f})  FPR={fpr:.4f}  '
          f'지연={lat:.4f}ms  감소율={red*100:5.1f}%  '
          f'허용={"O" if ok_mcc and ok_fpr else "X"}  성공={"O" if success else "X"}', flush=True)

# ---- 선택: 허용 한도 통과 + 성공 기준 충족 중, 목표축 개선폭(지연 감소) 큰 순 ----
eligible = [r for r in rows if r['pass_allowance'] and r['success']]
if eligible:
    eligible.sort(key=lambda r: (-r['lat_reduction'], -r['MCC']))   # 타이브레이크: 개선폭 -> 부작용
    chosen = eligible[0]
    print(f'\n선택: n_estimators={chosen["n_estimators"]} '
          f'(지연 {chosen["lat_reduction"]*100:.1f}% 감소, ΔMCC {chosen["dMCC"]:+.4f})')
else:
    chosen = None
    print('\n허용 한도·성공 기준을 모두 만족하는 후보 없음 → 개선 실패로 보고 (§10.2)')

json.dump({'baseline': dict(MCC=B_MCC, FPR=B_FPR, lat_ms=B_LAT),
           'candidates': rows, 'chosen': chosen},
          open(OUT + 'step1_lightweight.json', 'w'), indent=2)
