"""Phase 3 - GP 기반 베이지안 최적화 (Expected Improvement)
프로토콜 v2.6 §4.2 탐색공간 유지, 알고리즘만 TPE -> GP-EI로 대체
목적함수: Train 내부 3-fold CV의 MCC (Test 미사용)
"""
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel


class GPBayesOpt:
    """연속/정수/범주 혼합 탐색공간에 대한 GP-EI 베이지안 최적화.

    space: dict[name] = ('int', lo, hi) | ('cat', [values])
    모든 차원을 [0,1]로 정규화해 GP를 적합하고, EI를 최대화하는 점을 다음 시행으로 선택.
    """

    def __init__(self, space, n_init=10, n_iter=20, seed=42):
        self.space = space
        self.names = list(space.keys())
        self.n_init = n_init
        self.n_iter = n_iter
        self.rng = np.random.RandomState(seed)
        self.X, self.y = [], []

    # --- 정규화 좌표 <-> 실제 하이퍼파라미터 ---
    def _decode(self, u):
        p = {}
        for k, ui in zip(self.names, u):
            spec = self.space[k]
            if spec[0] == 'int':
                lo, hi = spec[1], spec[2]
                p[k] = int(np.clip(np.floor(lo + ui * (hi - lo + 1)), lo, hi))
            else:  # cat
                vals = spec[1]
                p[k] = vals[int(np.clip(np.floor(ui * len(vals)), 0, len(vals) - 1))]
        return p

    def _sample_u(self, n):
        return self.rng.rand(n, len(self.names))

    # --- Expected Improvement ---
    def _ei(self, gp, U, best, xi=0.01):
        mu, sd = gp.predict(U, return_std=True)
        sd = np.maximum(sd, 1e-9)
        imp = mu - best - xi
        z = imp / sd
        return imp * norm.cdf(z) + sd * norm.pdf(z)

    def run(self, objective, verbose=True):
        """objective(params) -> 점수(클수록 좋음). 총 n_init + n_iter회 평가."""
        # 1) 초기 무작위 설계
        for i in range(self.n_init):
            u = self._sample_u(1)[0]
            p = self._decode(u)
            s = objective(p)
            self.X.append(u); self.y.append(s)
            if verbose:
                print(f'  [init {i+1}/{self.n_init}] MCC={s:.4f} {p}', flush=True)

        # 2) GP-EI 반복
        kernel = ConstantKernel(1.0) * Matern(
            length_scale=np.ones(len(self.names)), nu=2.5) + WhiteKernel(1e-4)
        for i in range(self.n_iter):
            gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                          n_restarts_optimizer=2,
                                          random_state=self.rng.randint(1e6))
            gp.fit(np.array(self.X), np.array(self.y))
            # EI를 무작위 후보 다수에서 최대화
            cand = self._sample_u(2000)
            ei = self._ei(gp, cand, best=max(self.y))
            u = cand[int(np.argmax(ei))]
            p = self._decode(u)
            s = objective(p)
            self.X.append(u); self.y.append(s)
            if verbose:
                print(f'  [bo {i+1}/{self.n_iter}] MCC={s:.4f} best={max(self.y):.4f} {p}', flush=True)

        b = int(np.argmax(self.y))
        return self._decode(self.X[b]), self.y[b]
