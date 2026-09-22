import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':10.5,'axes.unicode_minus':False})
D = json.load(open('h1_data.json'))
M = ['A_RF','B_RF_BO','C_STACK','D_MLP']
LAB = {'A_RF':'A: RF 기본','B_RF_BO':'B: RF+BO','C_STACK':'C: 스태킹','D_MLP':'D: MLP'}
COL = {'A_RF':'#4C72B0','B_RF_BO':'#DD8452','C_STACK':'#55A868','D_MLP':'#8172B3'}

# ================= 그림 3: H3 포레스트 플롯 =================
eff = [('튜닝 효과\n(A → B)', 0.0097, 0.0069, 0.0124),
       ('스태킹 효과\n(B → C)', -0.0017, -0.0045, 0.0011),
       ('구조 차이\n(D → A)', 0.1979, 0.1886, 0.2071)]
fig, ax = plt.subplots(figsize=(7.6, 3.4))
for i, (n, e, lo, hi) in enumerate(eff):
    y = len(eff) - 1 - i
    c = '#999' if lo <= 0 <= hi else '#222'
    ax.errorbar(e, y, xerr=[[e-lo], [hi-e]], fmt='o', color=c, capsize=4, ms=7, lw=1.8)
    ax.text(hi + 0.006, y, f'{e:+.4f}  [{lo:+.4f}, {hi:+.4f}]', va='center', fontsize=9, color=c)
ax.axvline(0, color='k', lw=0.8)
ax.axvspan(-0.05, 0.05, color='#EEE', zorder=0)
ax.text(-0.048, 2.62, '실질적 차이 기준 ±0.05', ha='left', fontsize=8.5, color='#777')
ax.set_yticks(range(len(eff))); ax.set_yticklabels([e[0] for e in eff][::-1])
ax.set_xlabel('MCC 차이 (대응 비교, 시드 5회, 95% CI)')
ax.set_xlim(-0.06, 0.30); ax.set_ylim(-0.6, 2.9)
ax.spines[['top','right','left']].set_visible(False)
ax.tick_params(axis='y', length=0)
fig.tight_layout(); fig.savefig('fig/fig3_h3_forest.png', dpi=200); plt.close()

# ================= 그림 4: H1 4축 소형 다중 그림 =================
panels = [('탐지 MCC', 'mcc', 'mcc_ci', '↑ 높을수록 좋음', False),
          ('프라이버시: MIA AUC_adv', 'mia', 'mia_ci', '↓ 낮을수록 좋음 (0.5 = 무작위)', False),
          ('강건성: 최대 flip rate (20% 감소)', 'flip', 'flip_ci', '↓ 낮을수록 좋음', False),
          ('자원: 추론 지연 (ms/샘플, batch=10)', 'lat', None, '↓ 낮을수록 좋음 (로그 척도)', True)]
fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.5), sharey=True)
ys = np.arange(len(M))[::-1]
for ax, (title, k, kci, note, logx) in zip(axes, panels):
    for y, m in zip(ys, M):
        v = D[m][k]
        if kci:
            lo, hi = D[m][kci]
            ax.errorbar(v, y, xerr=[[v-lo], [hi-v]], fmt='o', color=COL[m], capsize=3.5, ms=7, lw=1.6)
        else:
            ax.plot(v, y, 'o', color=COL[m], ms=7)
            ax.plot([v, D[m]['lat_p95']], [y, y], color=COL[m], lw=1.6)
    if k == 'mia': ax.axvline(0.5, color='#999', ls='--', lw=0.9)
    if logx: ax.set_xscale('log')
    ax.set_title(title, fontsize=10.5, loc='left', pad=18)
    ax.text(0, 1.02, note, transform=ax.transAxes, fontsize=8.5, color='#666')
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='x', color='#EEE'); ax.set_axisbelow(True)
axes[0].set_yticks(ys); axes[0].set_yticklabels([LAB[m] for m in M])
axes[3].text(1.0, -0.2, '점 = p50, 선 = p50~p95 (1,000회 측정)', transform=axes[3].transAxes,
             ha='right', fontsize=8, color='#777')
fig.tight_layout(); fig.savefig('fig/fig4_h1_axes.png', dpi=200); plt.close()

# ================= 그림 5: H2 산점도 =================
pt = json.load(open('/home/claude/phase4/perturb.json'))
fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.6))
for ax, m in zip(axes, M):
    feats = pt[m]['H2']['features']
    xs = [pt[m]['importance'][f]['mean'] for f in feats]
    ys_ = [pt[m]['sensitivity'][f]['S_i'] for f in feats]
    for f, x, y in zip(feats, xs, ys_):
        hl = (f == 'net_connect')
        ax.scatter(x, y, s=55, color='#C0504D' if hl else COL[m], zorder=3,
                   edgecolor='k' if hl else 'none', linewidth=0.8)
        if hl or y > max(ys_) * 0.55:
            ax.annotate(f, (x, y), xytext=(5, 4), textcoords='offset points', fontsize=8,
                        color='#C0504D' if hl else '#333')
    rho = pt[m]['H2']['spearman_rho']; p = pt[m]['H2']['p_value']
    ax.set_title(f'{LAB[m]}\nρ = {rho:+.3f}  (p = {p:.3f}, n=7)', fontsize=10, loc='left')
    ax.set_xlabel('순열 중요도 (MCC 감소)', fontsize=9)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(color='#EEE'); ax.set_axisbelow(True)
axes[0].set_ylabel('교란 민감도 S_i', fontsize=9.5)
fig.text(0.5, -0.02, '※ 패널별 y축 척도가 다르다. H2는 모델 내부의 상관을 묻으므로 패널 간 크기 비교에 쓰지 않는다 '
         '(D의 민감도는 트리 모델의 약 1/8 수준).', ha='center', fontsize=8.5, color='#666')
fig.tight_layout(); fig.savefig('fig/fig5_h2_scatter.png', dpi=200, bbox_inches='tight'); plt.close()
print('ok')
