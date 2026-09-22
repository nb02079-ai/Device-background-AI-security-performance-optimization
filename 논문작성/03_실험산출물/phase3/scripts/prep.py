"""Phase 3 (재실행) - 데이터 준비. 프로토콜 v2.8 §2.1 / §3.4
공식 train 40,000 -> Train 60% / Validation 20% / H5-Test 20%
공식 test 24,747 -> 시간적 일반화 보고 전용 (봉인)
"""
import pandas as pd, numpy as np, json, hashlib, os
from sklearn.model_selection import train_test_split

SEED_SPLIT = 42
ROOT = '/mnt/user-data/uploads/'
OUT = '/home/claude/phase3/'
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(ROOT + 'dmbd_event_counts.csv')

# --- 2. 제외 특징 제거 (§3.3.2) ---
EXCLUDE = ['detonation', 'dur_after_detonation_s']
META = ['experiment', 't_first', 't_last', 't_detonate', 'year', 'label', 'test']
FEATURES = [c for c in df.columns if c not in EXCLUDE + META]
y = (df['label'] == 'malicious').astype(int)

# --- 3. 공식 분할 보존 + 공식 train을 3분할 (§2.1) ---
is_test = df['test'].astype(bool)
dev_idx = df.index[~is_test]        # 공식 train 40,000
off_test_idx = df.index[is_test]    # 공식 test 24,747 (봉인)

tr_idx, rest_idx = train_test_split(
    dev_idx, train_size=0.60, random_state=SEED_SPLIT, stratify=y.loc[dev_idx])
va_idx, h5_idx = train_test_split(
    rest_idx, train_size=0.50, random_state=SEED_SPLIT, stratify=y.loc[rest_idx])

print(f'Train      {len(tr_idx):6d}  악성 {y.loc[tr_idx].mean()*100:.1f}%')
print(f'Validation {len(va_idx):6d}  악성 {y.loc[va_idx].mean()*100:.1f}%')
print(f'H5-Test    {len(h5_idx):6d}  악성 {y.loc[h5_idx].mean()*100:.1f}%   [봉인]')
print(f'공식 Test  {len(off_test_idx):6d}  악성 {y.loc[off_test_idx].mean()*100:.1f}%   [봉인]')

# 봉인 셋 무결성 해시 (§9: 분할 시점에 인덱스를 고정하고 기록)
def idx_hash(idx):
    return hashlib.sha256(np.sort(np.asarray(idx)).tobytes()).hexdigest()[:16]

seal = {'H5_Test': idx_hash(h5_idx), 'official_Test': idx_hash(off_test_idx)}
print('\n봉인 해시:', seal)

# --- 4. Train에서만 중복 특징 판단·제거 (§3.3.2) ---
corr = df.loc[tr_idx, FEATURES].corr(method='spearman').abs()
drop_dup, pairs = [], []
for i, a in enumerate(FEATURES):
    if a in drop_dup: continue
    for b in FEATURES[i+1:]:
        if b in drop_dup: continue
        r = corr.loc[a, b]
        if r > 0.95:
            pairs.append((a, b, round(float(r), 4)))
            drop_dup.append(b)
FEAT_FINAL = [c for c in FEATURES if c not in drop_dup]

print('\n중복 쌍:', pairs if pairs else '없음')
print('제거:', drop_dup if drop_dup else '없음')
print('최종 특징', len(FEAT_FINAL), ':', FEAT_FINAL)

np.save(OUT+'tr_idx.npy', np.asarray(tr_idx))
np.save(OUT+'va_idx.npy', np.asarray(va_idx))
np.save(OUT+'h5_idx.npy', np.asarray(h5_idx))              # 봉인: Phase 6까지 로드 금지
np.save(OUT+'offtest_idx.npy', np.asarray(off_test_idx))   # 봉인
json.dump({'features_all': FEATURES, 'dropped_dup': drop_dup, 'dup_pairs': pairs,
           'features_final': FEAT_FINAL, 'seal_hashes': seal,
           'split': {'train': len(tr_idx), 'val': len(va_idx),
                     'h5_test': len(h5_idx), 'official_test': len(off_test_idx)}},
          open(OUT+'featureset.json','w'), ensure_ascii=False, indent=2)
df.to_pickle(OUT+'df.pkl')
print('\n저장 완료')
