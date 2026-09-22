"""상시 구동 관점 리소스 측정 (탐색적 부가 분석, Phase 6 판정에 영향 없음)

측정 항목
 ① 상주 메모리   : 모델 적재 전후 RSS 차이, 직렬화 크기, 트리 노드 총수
 ② 꼬리 지연     : 1,000회 반복의 p50/p95/p99/max (기존 §7은 20회 평균만)
 ③ 집계 비용     : 원시 이벤트 N건 -> 특징 벡터 1개 변환 비용

측정 불가 (한계로 명시)
 ④ CPU 경합      : 실행 환경이 단일 코어(cpu_count=1)여서 경합 실험이 측정 자체를 왜곡
 - 실제 Sysmon 후킹/수집 비용: OS 계층 문제이며 원시 로그 미보유

Train/Validation만 사용 (봉인 셋 미사용).
"""
import pandas as pd, numpy as np, json, time, pickle, gc, os, psutil, warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict

P3 = '/home/claude/phase3/'; OUT = '/home/claude/phase4/'
SEED = 42
N_TAIL = 1000          # 꼬리 지연용 반복
BATCH_TAIL = 10        # §10.1 기준 배치

df = pd.read_pickle(P3 + 'df.pkl')
FEAT = json.load(open(P3 + 'featureset.json'))['features_final']
BP = json.load(open(P3 + 'bo_result.json'))['best_params']
tr = np.load(P3 + 'tr_idx.npy'); va = np.load(P3 + 'va_idx.npy')
y = (df['label'] == 'malicious').astype(int).values
Xtr, ytr = df.loc[tr, FEAT].values, y[tr]
Xva = df.loc[va, FEAT].values
sc = StandardScaler().fit(Xtr)
proc = psutil.Process()

def rss_mb():
    gc.collect(); return proc.memory_info().rss / 1024**2

def n_nodes(m):
    """트리 계열의 총 노드 수 (모델 복잡도의 물리적 척도)"""
    if hasattr(m, 'estimators_'):
        return int(sum(e.tree_.node_count for e in m.estimators_))
    return None

results = {}

def measure(name, builder, infer_fn_factory):
    print(f'--- {name} ---', flush=True)
    # ① 상주 메모리: 적재 전후 RSS
    before = rss_mb()
    obj = builder()
    after = rss_mb()
    blob = pickle.dumps(obj)
    nodes = n_nodes(obj) if not isinstance(obj, tuple) else \
            sum(n_nodes(b) or 0 for b in obj[0])
    infer = infer_fn_factory(obj)

    # ② 꼬리 지연 (batch=10, 1000회)
    Xb = Xva[:BATCH_TAIL]
    for _ in range(20): infer(Xb)                     # 워밍업
    ts = np.empty(N_TAIL)
    for i in range(N_TAIL):
        t0 = time.perf_counter(); infer(Xb); ts[i] = time.perf_counter() - t0
    ms = ts * 1000 / BATCH_TAIL                        # 샘플당 ms
    r = dict(resident_mb=round(after - before, 2),
             pickle_mb=round(len(blob) / 1024**2, 2),
             tree_nodes=nodes,
             p50=float(np.percentile(ms, 50)), p95=float(np.percentile(ms, 95)),
             p99=float(np.percentile(ms, 99)), pmax=float(ms.max()),
             mean=float(ms.mean()))
    r['tail_ratio_p99_p50'] = round(r['p99'] / r['p50'], 2)
    results[name] = r
    print(f"  상주메모리 {r['resident_mb']:7.2f}MB  직렬화 {r['pickle_mb']:6.2f}MB  "
          f"노드수 {nodes if nodes else '-'}", flush=True)
    print(f"  지연(ms/샘플) p50={r['p50']:.4f} p95={r['p95']:.4f} "
          f"p99={r['p99']:.4f} max={r['pmax']:.4f}  p99/p50={r['tail_ratio_p99_p50']}", flush=True)
    del obj, blob; gc.collect()

measure('A_RF', lambda: RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=1).fit(Xtr, ytr),
        lambda m: (lambda X: m.predict_proba(X)[:, 1]))
measure('B_RF_BO', lambda: RandomForestClassifier(**BP, random_state=SEED, n_jobs=1).fit(Xtr, ytr),
        lambda m: (lambda X: m.predict_proba(X)[:, 1]))
measure('D_MLP', lambda: MLPClassifier(max_iter=200, early_stopping=True,
                                       n_iter_no_change=10, random_state=SEED).fit(sc.transform(Xtr), ytr),
        lambda m: (lambda X: m.predict_proba(sc.transform(X))[:, 1]))

def build_stack():
    bases = [RandomForestClassifier(**BP, random_state=SEED + k, n_jobs=1) for k in range(3)]
    bases.append(HistGradientBoostingClassifier(random_state=SEED))
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    oof = np.column_stack([cross_val_predict(b, Xtr, ytr, cv=cv, method='predict_proba', n_jobs=1)[:, 1]
                           for b in bases])
    meta = LogisticRegression(max_iter=1000).fit(oof, ytr)
    for b in bases: b.fit(Xtr, ytr)
    return (bases, meta)

measure('C_STACK', build_stack,
        lambda o: (lambda X: o[1].predict_proba(
            np.column_stack([b.predict_proba(X)[:, 1] for b in o[0]]))[:, 1]))

# ---------- ③ 집계 비용 ----------
print('\n--- ③ 집계 비용 (원시 이벤트 -> 특징 벡터) ---', flush=True)
EVENTS = ['process_create', 'process_term', 'image_load', 'net_connect', 'file_create',
          'reg_create_key', 'reg_write', 'reg_delete_key', 'reg_delete_value']
rng = np.random.RandomState(SEED)
agg = {}
for n_ev in [100, 680, 5000]:      # 680 = DMBD 실행당 평균 이벤트 수
    ev = pd.DataFrame({'RuleName': rng.choice(EVENTS, n_ev),
                       'timestamp': pd.to_datetime(rng.randint(0, 3*10**11, n_ev))})
    ts = []
    for _ in range(200):
        t0 = time.perf_counter()
        cnt = ev['RuleName'].value_counts()
        _ = [int(cnt.get(e, 0)) for e in EVENTS]
        _ = (ev['timestamp'].max() - ev['timestamp'].min()).total_seconds()
        ts.append(time.perf_counter() - t0)
    m = float(np.mean(ts) * 1000)
    agg[n_ev] = m
    print(f'  이벤트 {n_ev:5d}건 -> 벡터 1개: {m:.4f} ms', flush=True)
results['aggregation_ms'] = agg

json.dump(results, open(OUT + 'resource_extended.json', 'w'), indent=2)
print('\n=== 집계 비용 대비 추론 비용 (실행당 680 이벤트 기준) ===')
for m in ['A_RF', 'B_RF_BO', 'C_STACK', 'D_MLP']:
    inf1 = results[m]['p50']          # 샘플 1개 추론 (ms)
    print(f"  {m:10s} 집계 {agg[680]:.4f}ms vs 추론 {inf1:.4f}ms  -> 집계가 {agg[680]/inf1:6.1f}배")
