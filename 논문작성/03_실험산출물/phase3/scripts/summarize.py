"""Phase 3 결과 종합 — 프로토콜 v2.6 §8 (차이의 CI 기반 비교)"""
import numpy as np, json, pandas as pd
from sklearn.metrics import matthews_corrcoef

OUT='/home/claude/phase3/'
SEEDS=[42,7,123,2024,777]
res=json.load(open(OUT+'results_all.json'))

ORDER=['A_RF','B_RF_BO','C_STACK','D_MLP']
LABEL={'A_RF':'A: RF 기본','B_RF_BO':'B: RF+BO','C_STACK':'C: RF+BO+스태킹','D_MLP':'D: MLP'}
METRICS=['MCC','F1','Precision','Recall','PR_AUC','FPR']

print('=== Phase 3 Validation 성능 (시드 5회 평균 ± 표준편차, 임계값 0.5) ===\n')
rows=[]
for m in ORDER:
    runs=[res[m][str(s)] for s in SEEDS]
    row={'모델':LABEL[m]}
    for k in METRICS:
        v=np.array([r[k] for r in runs]); row[k]=f'{v.mean():.4f}±{v.std():.4f}'
    cm=np.array([[np.mean([r['TN'] for r in runs]),np.mean([r['FP'] for r in runs])],
                 [np.mean([r['FN'] for r in runs]),np.mean([r['TP'] for r in runs])]])
    row['평균혼동행렬']=f'[[{cm[0,0]:.0f},{cm[0,1]:.0f}],[{cm[1,0]:.0f},{cm[1,1]:.0f}]]'
    rows.append(row)
print(pd.DataFrame(rows).to_string(index=False))

# ---- 차이의 95% CI (대응 비교, 시드 짝지음) ----
print('\n\n=== 모델 간 MCC 차이의 95% CI (대응 비교, §8) ===')
print('  * 시드 5개 대응 t분포 기반. CI가 0을 포함하면 "유의미한 차이 발견 못함"\n')
from scipy import stats
def diff_ci(a,b):
    va=np.array([res[a][str(s)]['MCC'] for s in SEEDS])
    vb=np.array([res[b][str(s)]['MCC'] for s in SEEDS])
    d=va-vb; n=len(d)
    se=d.std(ddof=1)/np.sqrt(n); t=stats.t.ppf(0.975,n-1)
    return d.mean(), d.mean()-t*se, d.mean()+t*se

pairs=[('B_RF_BO','A_RF','튜닝 효과 (A→B)'),
       ('C_STACK','B_RF_BO','스태킹 효과 (B→C)'),
       ('C_STACK','A_RF','튜닝+스태킹 (A→C)'),
       ('A_RF','D_MLP','구조 효과 (D→A)')]
for a,b,name in pairs:
    m,lo,hi=diff_ci(a,b)
    inc='0 포함 → 유의미한 차이 발견 못함' if lo<=0<=hi else '0 불포함 → 차이 관측됨'
    print(f'  {name:22s} Δ={m:+.4f}  95%CI [{lo:+.4f}, {hi:+.4f}]  {inc}')

# ---- 임계값: 0.5 vs Youden's J (§8) ----
print('\n\n=== 임계값 분석 (§8: Youden J 조정 시 부트스트랩 CI 병기 의무) ===')
df=pd.read_pickle(OUT+'df.pkl'); va=np.load(OUT+'va_idx.npy')
yva=(df.loc[va,'label']=='malicious').astype(int).values
rng=np.random.RandomState(42)
for m in ORDER:
    prob=np.load(OUT+f'prob_{m}_42.npy')
    from sklearn.metrics import roc_curve
    fpr,tpr,thr=roc_curve(yva,prob); j=tpr-fpr
    t_opt=thr[int(np.argmax(j))]
    # 부트스트랩 CI
    bs=[]
    for _ in range(500):
        i=rng.randint(0,len(yva),len(yva))
        f2,t2,th2=roc_curve(yva[i],prob[i]); bs.append(th2[int(np.argmax(t2-f2))])
    lo,hi=np.percentile(bs,[2.5,97.5]); width=hi-lo
    mcc05=matthews_corrcoef(yva,(prob>=0.5).astype(int))
    mccJ=matthews_corrcoef(yva,(prob>=t_opt).astype(int))
    verdict='불안정 → 0.5 고정 (CI 폭>0.2)' if width>0.2 else '안정'
    print(f'  {LABEL[m]:18s} Youden J={t_opt:.3f} CI[{lo:.3f},{hi:.3f}] 폭={width:.3f} | MCC 0.5={mcc05:.4f} J={mccJ:.4f} | {verdict}')
