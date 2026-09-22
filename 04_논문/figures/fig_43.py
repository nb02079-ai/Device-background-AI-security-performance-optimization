import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':10.5,'axes.unicode_minus':False})

p1 = json.load(open('/home/claude/phase6/h5_eval_part1.json'))['diff_sel_minus_batch']
p2 = json.load(open('/home/claude/phase6/h5_eval_part2_mia.json'))['diff_sel_minus_batch']
# (이름, 구분, Δ, CI, δ, 높을수록좋음?)
rows = [('탐지 MCC',      '부작용축', p1['MCC']['est'],          p1['MCC']['ci'],          0.05, True),
        ('지연 (상대)',   '목표축',   p1['latency_rel']['est'],  p1['latency_rel']['ci'],  0.20, False),
        ('최대 flip rate','부작용축', p1['max_flip']['est'],     p1['max_flip']['ci'],     0.05, False),
        ('MIA AUC_adv',   '부작용축', p2['est'],                 p2['ci'],                 0.02, False)]

def verdict(lo_n, hi_n, side):
    # 정규화 좌표(오른쪽=선택적 유리)에서 판정
    if side == '부작용축' and lo_n > 0: return '우위', '#2E7D32'
    if lo_n >= -1: return '비열등', '#2E7D32'
    if hi_n < -1:  return '열등', '#C62828'
    return '판정 불가', '#8A8A8A'

fig, ax = plt.subplots(figsize=(9.2, 3.9))
for i, (name, side, e, ci, d, hb) in enumerate(rows):
    y = len(rows) - 1 - i
    sgn = 1 if hb else -1                     # 오른쪽 = 선택적 유리로 방향 통일
    en = sgn * e / d
    a, b = sorted([sgn * ci[0] / d, sgn * ci[1] / d])
    v, c = verdict(a, b, side)
    ax.plot([a, b], [y, y], color=c, lw=5, solid_capstyle='butt')
    ax.plot(en, y, 'o', color='white', mec=c, mew=2, ms=8, zorder=3)
    ax.text(2.45, y, f'{v}', va='center', ha='left', fontsize=10.5, color=c, fontweight='bold')
    ax.text(2.45, y - 0.27, f'Δ={e:+.4f}, δ={d}', va='center', ha='left', fontsize=8, color='#777')
ax.axvline(0, color='k', lw=0.9)
ax.axvline(-1, color='#C62828', lw=1.2, ls='--')
ax.text(-1.05, 3.55, '비열등성 경계 (−δ)', ha='right', fontsize=8.5, color='#C62828')
ax.text(0.05, 3.55, '차이 없음', ha='left', fontsize=8.5, color='#333')
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([f'{r[0]}\n({r[1]})' for r in rows][::-1], fontsize=9.5)
ax.set_xlim(-3.8, 2.4); ax.set_ylim(-0.6, 3.8)
ax.set_xlabel('정규화 차이 (Δ / δ)      ←  일괄이 유리          선택적이 유리  →')
ax.spines[['top','right','left']].set_visible(False); ax.tick_params(axis='y', length=0)
fig.text(0.02, -0.04, '막대 = 95% 부트스트랩 CI (2,000회), 원 = 점추정. 낮을수록 좋은 축은 부호를 뒤집어 방향을 통일했다. '
         'H5 지지 조건: 목표축 비열등 AND 부작용축 1개 이상 우위 AND 나머지 부작용축 비열등.',
         fontsize=8, color='#666', wrap=True)
fig.tight_layout(); fig.savefig('fig/fig6_h5_noninferiority.png', dpi=200, bbox_inches='tight'); plt.close()
print('ok')
