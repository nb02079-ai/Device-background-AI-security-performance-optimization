import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':11,'axes.unicode_minus':False})
C_BEN, C_MAL = '#3B6EA5', '#C0504D'
d = pd.read_csv('/mnt/user-data/uploads/dmbd_event_counts.csv', usecols=['year','label'])
d = d[d.year.notna()]; d['year'] = d.year.astype(int)
def grp(y):
    if y <= 2016: return '≤2016'
    if y in (2018, 2019): return '2018–19'
    return str(y)
d['g'] = d.year.apply(grp)
order = ['≤2016','2017','2018–19','2020','2021','2022','2023','2024']
t = pd.crosstab(d.g, d.label).reindex(order).fillna(0)
tot = t.sum(axis=1); N = tot.sum()
pm = t['malicious'] / tot

fig, ax = plt.subplots(figsize=(8.6, 4.6))
x0 = 0; gap = N * 0.004
for g in order:
    w = tot[g]; hm = pm[g]
    ax.bar(x0, hm, width=w, align='edge', color=C_MAL, edgecolor='white', linewidth=0.8)
    ax.bar(x0, 1 - hm, width=w, bottom=hm, align='edge', color=C_BEN, edgecolor='white', linewidth=0.8)
    cx = x0 + w / 2
    big = w / N > 0.06; mid = 0.03 < w / N <= 0.06     # 큰 칸 / 중간 칸 / 미소 칸
    if big or mid:
        fs = 11 if big else 8.5
        if hm >= 0.08:
            ax.text(cx, hm / 2, f'{hm*100:.0f}%', ha='center', va='center',
                    color='white', fontweight='bold', fontsize=fs)
        ax.text(cx, -0.07, g, ha='center', va='top', fontsize=10.5 if big else 8.5)
        ax.text(cx, -0.15, f'n={int(w):,}', ha='center', va='top', fontsize=8.5 if big else 7.5, color='#555')
    x0 += w + gap
ax.set_xlim(0, x0); ax.set_ylim(0, 1)
ax.set_xticks([]); ax.set_yticks([0, .25, .5, .75, 1]); ax.set_yticklabels(['0%','25%','50%','75%','100%'])
ax.set_ylabel('클래스 구성 비율')
ax.spines[['top','right','bottom']].set_visible(False)
ax.text(0.5, -0.27, '막대 폭 ∝ 표본 수   (표본이 극히 적은 ≤2016(n=287), 2018–19(n=440)는 라벨 생략)',
        transform=ax.transAxes, ha='center', fontsize=9, color='#666')
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=C_MAL, label='악성'), Patch(color=C_BEN, label='정상')],
          frameon=False, ncol=2, loc='upper center', bbox_to_anchor=(0.5, 1.10))
fig.tight_layout(); fig.savefig('fig/fig2b_dmbd_mosaic.png', dpi=200, bbox_inches='tight'); plt.close()
print('ok')
