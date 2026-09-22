import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':10.5,'axes.unicode_minus':False})
R = json.load(open('/home/claude/phase6/official_test_cohort.json'))
M = ['A_RF','B_RF_BO','C_STACK','Selective','Batch','D_MLP']
LAB = {'A_RF':'A: RF 기본','B_RF_BO':'B: RF+BO','C_STACK':'C: 스태킹',
       'Selective':'선택적 개선','Batch':'일괄 개선','D_MLP':'D: MLP'}
COL = {'A_RF':'#4C72B0','B_RF_BO':'#DD8452','C_STACK':'#55A868',
       'Selective':'#937860','Batch':'#C44E52','D_MLP':'#8172B3'}
rec = {m: R[m]['2024_악성']['value'] for m in M}
auc = {m: R[m]['auc_2024'] for m in M}

# ================= 그림 7: 탐지율 vs AUC (가로·세로 눈금 간격 동일) =================
fig, ax = plt.subplots(figsize=(9.0, 5.0))
for m in M:
    ax.scatter(rec[m], auc[m], s=90, color=COL[m], zorder=3, edgecolor='white', linewidth=1)
    off = {'Batch':(-8,8,'right'), 'B_RF_BO':(-8,6,'right'), 'D_MLP':(8,6,'left'),
           'C_STACK':(6,6,'left'), 'A_RF':(6,-13,'left'), 'Selective':(-8,-14,'right')}[m]
    ax.annotate(LAB[m], (rec[m], auc[m]), xytext=off[:2], textcoords='offset points',
                ha=off[2], fontsize=9.5, color=COL[m])
rf = [m for m in M if m != 'D_MLP']
rlo, rhi = min(rec[m] for m in rf), max(rec[m] for m in rf)
alo, ahi = min(auc[m] for m in rf), max(auc[m] for m in rf)
yb = 0.9405
ax.annotate('', xy=(rlo, yb), xytext=(rhi, yb), arrowprops=dict(arrowstyle='<->', color='#333', lw=1.2))
ax.text((rlo+rhi)/2, yb+0.002, f'탐지율 폭 {rhi-rlo:.3f}', ha='center', va='bottom', fontsize=9.5)
xb = 0.905
ax.annotate('', xy=(xb, alo), xytext=(xb, ahi), arrowprops=dict(arrowstyle='<->', color='#333', lw=1.2))
ax.text(xb+0.003, (alo+ahi)/2, f'AUC 폭\n{ahi-alo:.3f}', ha='left', va='center', fontsize=9.5)
ax.set_xlim(0.69, 0.935); ax.set_ylim(0.835, 0.950)
ax.set_aspect('equal')                                    # 가로 0.01 = 세로 0.01
ax.set_xlabel('2024 악성 탐지율 (임계값 0.5)')
ax.set_ylabel('2024 코호트 ROC-AUC\n(임계값 비의존 구분 능력)')
ax.spines[['top','right']].set_visible(False)
ax.grid(color='#EEE'); ax.set_axisbelow(True)
fig.text(0.5, -0.03, '가로·세로 눈금 간격을 동일하게 고정했다 (0.01 = 0.01). 탐지율 폭과 AUC 폭은 D를 제외한 RF 계열 5개 모델 기준.',
         ha='center', fontsize=8.5, color='#666')
fig.tight_layout(); fig.savefig('fig/fig7_recall_vs_auc.png', dpi=200, bbox_inches='tight'); plt.close()

# ================= 그림 8: 평균 확률 덤벨 =================
order = sorted(M, key=lambda m: -auc[m])          # 그림 7과 같은 AUC 순
fig, ax = plt.subplots(figsize=(8.6, 4.2))
for i, m in enumerate(order):
    y = len(order) - 1 - i
    b, mm = R[m]['mean_prob_2024_ben'], R[m]['mean_prob_2024_mal']
    ax.plot([b, mm], [y, y], color='#BBB', lw=2.5, zorder=1)
    ax.scatter(b, y, s=80, color='#3B6EA5', zorder=3, label='2024 정상 평균확률' if i == 0 else None)
    ax.scatter(mm, y, s=80, color='#C0504D', zorder=3, label='2024 악성 평균확률' if i == 0 else None)
    ax.text(1.02, y, f'AUC {auc[m]:.4f}', va='center', fontsize=9, color='#555')
ax.axvline(0.5, color='k', ls='--', lw=0.9)
ax.text(0.5, len(order) - 0.35, '판정 임계값 0.5', ha='center', fontsize=8.5)
ax.set_yticks(range(len(order))); ax.set_yticklabels([LAB[m] for m in order][::-1])
ax.set_xlim(0, 1); ax.set_xlabel('모델이 산출한 평균 악성 확률')
ax.spines[['top','right','left']].set_visible(False); ax.tick_params(axis='y', length=0)
ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=2)
fig.text(0.5, -0.08, '점의 위치는 판단 경계(보정)를 나타낸다. 두 점 사이 거리는 구분 능력이 아니다 — 구분 능력은 그림 7의 AUC로 본다 '
         '(평균만으로는 두 분포의 겹침을 알 수 없음).', ha='center', fontsize=8.3, color='#666')
fig.tight_layout(); fig.savefig('fig/fig8_prob_dumbbell.png', dpi=200, bbox_inches='tight'); plt.close()
print('ok')
