"""Phase 4 - 리소스 벤치마크 (프로토콜 v2.9 §7)
- 전처리/추론 비용 분리
- batch = 집계된 특징벡터 수 (10 / 100 / 1,000)
- tracemalloc(Python 할당) + psutil RSS(프로세스 물리메모리) 병행
- 워밍업 1회, 단일스레드, 모델 사전 적재 상태에서 측정
- 각 조건 20회 반복
Test 미사용 (§9)
"""
import pandas as pd, numpy as np, json, time, tracemalloc, psutil, os, warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict

OUT = '/home/claude/phase4/'; os.makedirs(OUT, exist_ok=True)
P3 = '/home/claude/phase3/'
BATCHES = [10, 100, 1000]
N_REP = 20
SEED = 42

df = pd.read_pickle(P3 + 'df.pkl')
FEAT = json.load(open(P3 + 'featureset.json'))['features_final']
tr = np.load(P3 + 'tr_idx.npy'); va = np.load(P3 + 'va_idx.npy')
y = (df['label'] == 'malicious').astype(int).values
Xtr, ytr = df.loc[tr, FEAT].values, y[tr]
Xva = df.loc[va, FEAT].values
BP = json.load(open(P3 + 'bo_result.json'))['best_params']

# ---------- 모델 적재 (측정 전 완료) ----------
print('모델 학습 중...', flush=True)
sc = StandardScaler().fit(Xtr)

models = {}
models['A_RF'] = ('tree', RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=1).fit(Xtr, ytr))
models['B_RF_BO'] = ('tree', RandomForestClassifier(**BP, random_state=SEED, n_jobs=1).fit(Xtr, ytr))
models['D_MLP'] = ('mlp', MLPClassifier(max_iter=200, early_stopping=True,
                                        n_iter_no_change=10, random_state=SEED).fit(sc.transform(Xtr), ytr))

# 스태킹: 베이스 + 메타를 함께 적재
bases = [RandomForestClassifier(**BP, random_state=SEED + k, n_jobs=1) for k in range(3)]
bases.append(HistGradientBoostingClassifier(random_state=SEED))
oof_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.column_stack([cross_val_predict(b, Xtr, ytr, cv=oof_cv,
                                         method='predict_proba', n_jobs=1)[:, 1] for b in bases])
meta = LogisticRegression(max_iter=1000).fit(oof, ytr)
for b in bases: b.fit(Xtr, ytr)
models['C_STACK'] = ('stack', (bases, meta))
print('완료\n', flush=True)


def infer(kind, m, X):
    """추론 1회 (전처리 제외)"""
    if kind == 'tree':
        return m.predict_proba(X)[:, 1]
    if kind == 'mlp':
        return m.predict_proba(X)[:, 1]
    bases, meta = m
    return meta.predict_proba(np.column_stack([b.predict_proba(X)[:, 1] for b in bases]))[:, 1]


def preprocess(kind, X):
    """전처리 비용 (MLP만 스케일링 필요)"""
    return sc.transform(X) if kind == 'mlp' else X


proc = psutil.Process()
results = {}

for name, (kind, m) in models.items():
    results[name] = {}
    for B in BATCHES:
        Xb_raw = Xva[:B]
        Xb = preprocess(kind, Xb_raw)

        infer(kind, m, Xb)            # 워밍업 1회

        # --- 추론 지연 ---
        t_inf = []
        for _ in range(N_REP):
            t0 = time.perf_counter(); infer(kind, m, Xb); t_inf.append(time.perf_counter() - t0)
        # --- 전처리 비용 ---
        t_pre = []
        for _ in range(N_REP):
            t0 = time.perf_counter(); preprocess(kind, Xb_raw); t_pre.append(time.perf_counter() - t0)

        # --- 메모리 ---
        tracemalloc.start()
        infer(kind, m, Xb)
        _, peak_py = tracemalloc.get_traced_memory(); tracemalloc.stop()
        rss_before = proc.memory_info().rss
        infer(kind, m, Xb)
        rss_after = proc.memory_info().rss

        ti = np.array(t_inf) * 1000    # ms
        tp = np.array(t_pre) * 1000
        results[name][B] = dict(
            infer_ms_mean=float(ti.mean()), infer_ms_std=float(ti.std()),
            infer_ms_per_sample=float(ti.mean() / B),
            pre_ms_mean=float(tp.mean()), pre_ms_per_sample=float(tp.mean() / B),
            throughput_per_s=float(B / (ti.mean() / 1000)),
            tracemalloc_peak_kb=peak_py / 1024,
            rss_delta_kb=(rss_after - rss_before) / 1024)
        print(f'{name:10s} batch={B:5d}  추론 {ti.mean():8.3f}ms '
              f'({ti.mean()/B:.5f}ms/샘플)  전처리 {tp.mean():.4f}ms  '
              f'처리량 {B/(ti.mean()/1000):9.0f}/s  py-peak {peak_py/1024:7.1f}KB', flush=True)
    print()

json.dump(results, open(OUT + 'resource.json', 'w'), indent=2)

# 환경 정보 기록 (§7 재현성)
env = dict(cpu_count=psutil.cpu_count(logical=True),
           total_mem_gb=round(psutil.virtual_memory().total / 1024**3, 1),
           n_jobs='1 (단일스레드 고정)')
json.dump(env, open(OUT + 'env.json', 'w'), indent=2)
print('측정 환경:', env)
