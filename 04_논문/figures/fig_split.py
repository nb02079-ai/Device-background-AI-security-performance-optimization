import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':10,'axes.unicode_minus':False})

fig, ax = plt.subplots(figsize=(12.5, 6.6))
ax.set_xlim(0, 12.5); ax.set_ylim(-0.1, 6.6); ax.axis('off')

def box(x, y, w, h, fc, ec, title, lines, tcol='#111', lock=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.12',
                                fc=fc, ec=ec, lw=1.6))
    ax.text(x + 0.15, y + h - 0.22, title + ('  [봉인]' if lock else ''), fontsize=10.5,
            fontweight='bold', va='top', color=tcol)
    for i, ln in enumerate(lines):
        ax.text(x + 0.15, y + h - 0.58 - i * 0.28, ln, fontsize=8.6, va='top', color='#333')

def arrow(x1, y1, x2, y2, col='#666'):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=13,
                                 color=col, lw=1.3, connectionstyle='arc3,rad=0'))

# 0단: 원천
box(0.15, 2.55, 2.1, 1.5, '#F2F2F2', '#888', 'DMBD 2025',
    ['64,747 실행', '(이벤트 0건 669 제외)', '악성 61.0%'])

# 1단: 공식 분할
box(3.0, 3.75, 2.35, 1.55, '#EAF1F8', '#3B6EA5', '공식 train',
    ['40,000 실행', '악성 50.0%', '연도 혼합'])
box(3.0, 1.0, 2.35, 1.95, '#FBEAEA', '#C0504D', '공식 test',
    ['24,747 실행', '악성 78.8% ⚠', '2017 악성 17,131', '2024 악성 2,367 / 정상 5,248'], lock=True)
arrow(2.25, 3.55, 3.0, 4.35); arrow(2.25, 3.05, 3.0, 2.2)

# 2단: 재분할 (공식 train -> 3분할)
box(6.15, 5.05, 2.55, 1.3, '#FFFFFF', '#3B6EA5', 'Train  24,000',
    ['60% · 악성 50.0%', '모델 학습 · BO 내부 3-fold CV'])
box(6.15, 3.45, 2.55, 1.3, '#FFFFFF', '#3B6EA5', 'Validation  8,000',
    ['20% · 악성 50.0%', '진단 · 개선안 선택 · 임계값'])
box(6.15, 1.85, 2.55, 1.3, '#FFF4E0', '#D68910', 'H5-Test  8,000',
    ['20% · 악성 50.0%', 'hash 47254fd15bb1f192'], lock=True)
for yy in (5.7, 4.1, 2.5):
    arrow(5.35, 4.5, 6.15, yy, col='#3B6EA5')
ax.text(5.72, 5.35, '층화\n3분할', fontsize=8, ha='center', color='#3B6EA5')

# 3단: 용도
use = [(5.7, '#3B6EA5', 'Phase 3  모델 4종 학습'),
       (4.1, '#3B6EA5', 'Phase 4  3대 진단  ·  Phase 6  개선안 선택'),
       (2.5, '#D68910', 'Phase 6  H5 최종 판정 (1회만)'),
       (1.30, '#C0504D', 'Phase 6  시간적 일반화 보고 (코호트별)')]
for (yy, c, t), src in zip(use, [(8.7, 5.7), (8.7, 4.1), (8.7, 2.5), (5.35, 1.30)]):
    arrow(src[0], src[1], 9.25, yy, col=c)
    ax.text(9.35, yy, t, fontsize=9.2, va='center', color=c, fontweight='bold')
ax.text(9.35, 0.98, 'hash c63d8ca03e7a4480 · H5 판정에 사용 금지', fontsize=8, va='center', color='#C0504D')

# 원칙
ax.text(0.15, 0.40, '선택과 검증의 분리:', fontsize=9.2, fontweight='bold', color='#111')
ax.text(2.0, 0.40, 'Validation에서 개선안을 고르고, 한 번도 열람하지 않은 H5-Test에서 판정한다.  '
        '[봉인] 셋은 해시 검증 후 1회 해제.', fontsize=9, color='#333')
ax.text(0.15, 0.06, '⚠ 공식 test는 연도-라벨 교란(test 정상은 전부 2024년)과 사전확률 이동(50%→78.8%)이 있어 H5 판정 무대에서 제외했다.',
        fontsize=8.6, color='#C0504D')
fig.savefig('fig/fig_split_design.png', dpi=200, bbox_inches='tight'); plt.close()
print('ok')
