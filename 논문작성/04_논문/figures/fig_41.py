import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':11,'axes.unicode_minus':False})
C_BEN, C_MAL = '#3B6EA5', '#C0504D'

# ---------- 그림 1: CIC-MalMem svcscan.nservices ----------
df = pd.read_csv('/mnt/user-data/uploads/MalMem2022.csv', usecols=['svcscan.nservices','Class'])
b = df[df.Class=='Benign']['svcscan.nservices']; m = df[df.Class=='Malware']['svcscan.nservices']
xs = np.arange(376, 397)
bc = np.array([(b==x).sum() for x in xs]); mc = np.array([(m==x).sum() for x in xs])
fig, ax = plt.subplots(figsize=(8.2, 4.2))
w = 0.42
ax.bar(xs - w/2, bc, w, color=C_BEN, label=f'정상 (n={len(b):,})')
ax.bar(xs + w/2, mc, w, color=C_MAL, label=f'악성 (n={len(m):,})')
for x, v, c in [(395, (b==395).sum(), C_BEN), (389, (m==389).sum(), C_MAL)]:
    ax.annotate(f'{v:,}건', (x + (-w/2 if c==C_BEN else w/2), v), ha='center', va='bottom',
                fontsize=10, color=c, fontweight='bold')
ax.set_xlabel('실행 중인 서비스 수 (svcscan.nservices)')
ax.set_ylabel('메모리 덤프 수')
ax.set_xticks(xs[::2])
ax.spines[['top','right']].set_visible(False)
ax.legend(frameon=False, loc='upper left')
ax.text(0.99, 0.97, f'표시 범위 밖(376 미만): 악성 {(m<376).sum()}건',
        transform=ax.transAxes, ha='right', va='top', fontsize=9, color='#666')
fig.tight_layout(); fig.savefig('fig/fig1_cic_nservices.png', dpi=200); plt.close()

# ---------- 그림 2: DMBD 연도별 클래스 구성 ----------
d = pd.read_csv('/mnt/user-data/uploads/dmbd_event_counts.csv', usecols=['year','label'])
d = d[d.year.notna()]; d['year'] = d.year.astype(int)
def grp(y):
    if y <= 2016: return '≤2016'
    if y in (2018, 2019): return '2018–19'
    return str(y)
d['g'] = d.year.apply(grp)
order = ['≤2016','2017','2018–19','2020','2021','2022','2023','2024']
t = pd.crosstab(d.g, d.label).reindex(order).fillna(0)
tot = t.sum(1); pm = t['malicious']/tot*100; pb = t['benign']/tot*100
fig, ax = plt.subplots(figsize=(8.2, 4.4))
x = np.arange(len(order))
ax.bar(x, pm, color=C_MAL, label='악성')
ax.bar(x, pb, bottom=pm, color=C_BEN, label='정상')
for i, (n, p) in enumerate(zip(tot, pm)):
    ax.text(i, 102, f'n={int(n):,}', ha='center', va='bottom', fontsize=8.5, color='#444')
    if p >= 8:
        ax.text(i, p/2, f'{p:.0f}%', ha='center', va='center', fontsize=9, color='white', fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(order)
ax.set_ylabel('클래스 구성 비율 (%)'); ax.set_xlabel('샘플 최초 발견 연도')
ax.set_ylim(0, 112)
ax.spines[['top','right']].set_visible(False)
ax.legend(frameon=False, ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.16))
fig.tight_layout(); fig.savefig('fig/fig2_dmbd_year_label.png', dpi=200); plt.close()
print('ok')
