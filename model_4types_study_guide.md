# 모델 4종 학습 자료 정리 (프로토콜 v1.8 §4.1)

**작성일**: 2026-09-15
**대상**: A(RandomForest 기본) / B(RF+베이지안최적화) / C(RF+BO+스태킹) / D(MLP)
**목적**: 각 모델의 원리를 이해하고, 왜 이 4종을 이 순서로 배치했는지 파악한다.

---

## 0. 먼저: 왜 이 4종인가

이 구성은 "어느 모델이 제일 좋은가"를 묻는 게 아니다. **차이를 하나씩만 바꿔서 그 차이의 효과를 분리**하려는 설계다.

```
A (RF 기본)
 │  ← 이 차이 = 튜닝 방법의 효과
B (RF + 베이지안 최적화)
 │  ← 이 차이 = 앙상블 결합 구조의 효과
C (RF + BO + 스태킹)

A/B/C  vs  D (MLP)
 └─ 이 차이 = 모델 구조(트리 vs 신경망)의 효과
```

각 단계가 **직전 단계에 한 가지만 더한 것**이므로, 성능·프라이버시·강건성 차이가 나타나면 그 원인을 특정할 수 있다. 이런 설계를 실험설계에서 **절제 연구(ablation study)**라 부른다.

> 주의: v1.8 프로토콜은 D를 "순수 구조 효과"로 부르지 말라고 명시한다. MLP와 RF는 탐색 예산·정규화·불균형 처리 조건이 다를 수 있어, 엄밀히는 "지정된 학습 조건 하의 파이프라인 차이"다.

---

## 1. Random Forest (모델 A) — 기초

### 1-1. 원전 논문
**Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5-32.**
- https://link.springer.com/article/10.1023/A:1010933404324
- 저자 원본 PDF: https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf

### 1-2. 핵심 아이디어 (읽기 전 개념 정리)

의사결정나무 하나는 학습 데이터에 과하게 맞춰지는(과적합) 성질이 강하다. RF는 이를 **두 겹의 무작위성**으로 해결한다:

1. **배깅(Bagging)**: 학습 데이터에서 복원추출로 여러 부분집합을 만들어 각각 다른 나무를 학습
2. **특징 무작위 선택**: 각 노드에서 분기할 때 전체 특징이 아니라 **무작위로 뽑은 일부 특징**만 후보로 고려

두 번째가 Breiman의 핵심 기여다. 배깅만 하면 나무들이 서로 비슷해지는데(강한 특징 하나를 모두가 먼저 쓰므로), 특징을 무작위로 제한하면 **나무들이 서로 달라져서(decorrelated)** 평균 낼 때 분산 감소 효과가 커진다.

### 1-3. 우리 연구에서 알아야 할 하이퍼파라미터

| 이름 | 의미 | 프로토콜 4.2절 탐색 범위 |
|---|---|---|
| `n_estimators` | 나무 개수 | [10, 200] (기본 300) |
| `max_depth` | 나무 최대 깊이 | [5, 50] |
| `max_features` | 각 노드에서 고려할 특징 수 | [1, 20] ← **Breiman의 핵심 아이디어에 해당** |
| `min_samples_split` | 분기에 필요한 최소 샘플 수 | [2, 11] |
| `min_samples_leaf` | 리프 노드 최소 샘플 수 | [1, 11] |
| `criterion` | 분기 품질 측정 기준 | {gini, entropy} |

> **깊이와 리프 크기가 프라이버시와 직결된다**: 깊이 제한이 없고 리프가 작으면 나무가 개별 학습 샘플을 거의 암기하게 되어, 5장 멤버십 추론 공격에 취약해진다. v1.8 프로토콜 10.1절의 프라이버시 개선안이 "정규화 강화"인 이유가 이것이다.

### 1-4. 실무 문서
- scikit-learn 사용자 가이드 (Ensembles): https://scikit-learn.org/stable/modules/ensemble.html
- `RandomForestClassifier` API: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

### 1-5. 특징 중요도 — 주의할 점 ⭐
RF는 `feature_importances_`를 제공하지만 이것은 **불순도 감소 기반(MDI)**이며 알려진 편향이 있다: 카디널리티가 높은(값 종류가 많은) 특징을 과대평가하고, **학습 데이터 기준**이라 과적합의 영향을 받는다.

→ 그래서 프로토콜 5.1.1절에서 **순열 중요도(Permutation Importance)**를 주 방법으로 쓰기로 한 것이다.
- 공식 문서 (두 방법의 차이를 명시적으로 비교): https://scikit-learn.org/stable/modules/permutation_importance.html

---

## 2. 베이지안 최적화 (모델 B) — 하이퍼파라미터 튜닝

### 2-1. 무엇을 해결하는가

하이퍼파라미터 조합을 찾는 방법은 셋이다:

| 방법 | 원리 | 문제 |
|---|---|---|
| 그리드 서치 | 격자의 모든 조합 시도 | 차원이 늘면 폭발적 증가 |
| 랜덤 서치 | 무작위 조합 시도 | 과거 시도에서 배우지 않음 |
| **베이지안 최적화** | **과거 시도 결과로 "다음에 어디를 시도할지" 추론** | 구현 복잡, 순차적이라 병렬화 제약 |

핵심은 **각 시도가 비싸다**는 것이다(모델 학습에 수 분~수 시간). 그러니 시도 횟수를 줄여야 하고, 그러려면 과거 결과를 활용해야 한다.

### 2-2. 필독 자료 ⭐
**Frazier, P. I. (2018). A Tutorial on Bayesian Optimization. arXiv:1807.02811**
- https://arxiv.org/abs/1807.02811

이 분야의 표준 입문서다. 가우시안 프로세스 회귀와 세 가지 획득 함수(expected improvement, entropy search, knowledge gradient)를 설명한다.

> 이 튜토리얼에는 우리에게 중요한 경험칙이 하나 있다: **차원 수를 20 미만으로 유지할 것.** 우리 RF 탐색 공간은 6개 하이퍼파라미터이므로 안전 범위다.

### 2-3. TPE — 우리가 실제로 쓸 알고리즘
**Bergstra, J., Bardenet, R., Bengio, Y., & Kégl, B. (2011). Algorithms for Hyper-Parameter Optimization. NIPS 24, 2546-2554.**
- PDF: http://papers.neurips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf

**TPE(Tree-structured Parzen Estimator)가 일반 베이지안 최적화와 다른 점**:
- 표준 BO: 목적함수를 파라미터의 함수로 모델링 → `p(y|x)`
- TPE: **성능을 조건으로 파라미터 분포를 모델링** → `p(x|y)`

구체적으로 TPE는 관측치를 성능 기준으로 "좋음"(임계값 y* 이하)과 "나쁨" 두 그룹으로 나누고, 각 그룹의 파라미터 분포를 따로 추정한 뒤, **"좋음 그룹에서 나올 확률이 높고 나쁨 그룹에서 나올 확률이 낮은" 지점**을 다음 시도로 고른다.

> 왜 TPE인가: IDS-ML(Yang et al.)이 hyperopt의 TPE를 사용했고, 우리는 그 탐색 공간을 채택했다. 다만 **목적함수는 그들과 달리** Train 내부 nested CV로 계산한다(v1.8 §4.2.1 참고).

### 2-4. 함께 읽으면 좋은 것
**Bergstra, J., & Bengio, Y. (2012). Random Search for Hyper-Parameter Optimization. JMLR 13, 281-305.**
- https://jmlr.org/papers/v13/bergstra12a.html
- "왜 그리드 서치보다 랜덤 서치가 나은가"를 보인 유명한 논문. BO를 이해하기 전 단계로 좋다.

### 2-5. 구현
- hyperopt 문서: http://hyperopt.github.io/hyperopt/
- Optuna(TPE 기본 탑재, 더 현대적): https://optuna.readthedocs.io/

---

## 3. 스태킹 (모델 C) — 앙상블 결합

### 3-1. 원전
**Wolpert, D. H. (1992). Stacked Generalization. *Neural Networks*, 5(2), 241-259.**

### 3-2. 핵심 아이디어

여러 모델(**베이스 학습기**)의 예측을 **또 다른 모델(메타 학습기)의 입력으로** 쓴다. 투표나 평균처럼 고정된 규칙으로 합치는 대신, **"어떤 상황에서 어느 베이스 모델을 믿을지"를 학습**하는 것이다.

```
입력 x
 ├→ 베이스모델 1 → 예측 p1 ┐
 ├→ 베이스모델 2 → 예측 p2 ├→ 메타모델 → 최종 예측
 └→ 베이스모델 3 → 예측 p3 ┘
```

### 3-3. 가장 중요한 함정: OOF ⭐⭐

**메타모델을 무엇으로 학습시킬 것인가**가 스태킹의 핵심이자 가장 흔한 실수 지점이다.

**틀린 방법** (IDS-ML이 실제로 이렇게 했다 — v1.8 §4.2.1):
```python
p_train = base_model.predict(X_train)   # 자기가 학습한 데이터를 예측
meta_model.fit(p_train, y_train)
```
베이스 모델은 학습 데이터를 거의 완벽히 맞히므로, 메타모델은 **"베이스 모델은 항상 옳다"**는 비현실적 패턴을 학습한다. 실제 배포 환경에서 성능이 떨어진다.

**올바른 방법 (OOF, Out-Of-Fold)**:
```
Train을 5-fold로 나눔
 fold 1을 제외하고 학습 → fold 1을 예측 → 이 예측을 저장
 fold 2를 제외하고 학습 → fold 2를 예측 → 이 예측을 저장
 ...
 저장된 예측(= 각 샘플이 "학습에 안 쓰인 상태"에서 받은 예측)으로 메타모델 학습
```

이렇게 하면 메타모델이 보는 예측이 **실제 배포 시 마주칠 예측 품질**과 같아진다.

- scikit-learn `StackingClassifier`(내부적으로 교차검증 예측 사용): https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html
- 앙상블 가이드의 스태킹 절: https://scikit-learn.org/stable/modules/ensemble.html#stacked-generalization

### 3-4. 우리 구성 (프로토콜 4.2절)
- 베이스: RF 3종(서로 다른 시드) + Gradient Boosting 1종
- 메타: 로지스틱 회귀
- OOF 분할: Train 내 5-fold

> 메타모델을 로지스틱 회귀 같은 단순한 모델로 두는 것이 일반적이다. 메타모델까지 복잡하면 2차 과적합이 생긴다.

---

## 4. MLP (모델 D) — 신경망 대조군

### 4-1. 개념

다층 퍼셉트론(Multi-Layer Perceptron). 입력층 → 은닉층(1개 이상) → 출력층으로 이어지는 가장 기본적인 신경망. 각 층은 선형 변환 후 비선형 활성함수(ReLU 등)를 거친다.

트리 모델과의 근본적 차이:

| | 트리 계열 | MLP |
|---|---|---|
| 결정 경계 | 축에 평행한 계단 형태 | 매끄러운 곡면 |
| 입력 스케일 | **영향 없음** | **정규화 필수** |
| 해석 | 분기 규칙 추적 가능 | 어려움 |
| 특징 상호작용 | 분기로 표현 | 층을 통해 자동 학습 |

### 4-2. 우리 실험에서 반드시 챙길 것 ⭐

**스케일링이 필수다.** 트리는 "CPU 사용률 > 30" 같은 임계값 비교만 하므로 단위가 무관하지만, MLP는 가중치 곱셈이라 **값의 크기 차이가 학습을 망친다.** 예컨대 메모리(수백 MB)와 네트워크 연결 수(한 자리)를 그대로 넣으면 메모리 특징이 학습을 지배한다.

→ 프로토콜 2.3절: 스케일러는 **Train에서만 fit**하고 Val/Test에는 transform만 적용 (누수 방지)

### 4-3. 프로토콜 4.2절 설정
- 은닉층 1~2개, 유닛 수 [16, 128]
- 조기종료: Validation loss가 10 epoch 정체 시 중단
- 최대 200 epoch

> 표본이 작을 때(우리 3.2절 결정으로 실행 단위 집계 후 표본 감소) **작은 네트워크로 가야 한다.** 큰 네트워크는 과적합되고 학습이 불안정해진다.

### 4-4. 문서
- scikit-learn 신경망 가이드: https://scikit-learn.org/stable/modules/neural_networks_supervised.html
- `MLPClassifier` API: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html

---

## 5. 전체를 관통하는 교과서

**Hastie, T., Tibshirani, R., & Friedman, J. — The Elements of Statistical Learning (2nd ed.)**
- 무료 PDF: https://hastie.su.domains/ElemStatLearn/
- 관련 장: 9장(트리), 11장(신경망), 15장(랜덤 포레스트), 16장(앙상블)

**James, G. et al. — An Introduction to Statistical Learning**
- 무료 PDF: https://www.statlearning.com/
- 위 책의 입문판. 수학 부담이 적고 실습 코드가 있다. **기계공학 배경에서 통계학습에 진입할 때 이쪽이 낫다.**

---

## 6. 추천 학습 순서

| 순서 | 자료 | 소요 | 이유 |
|---|---|---|---|
| 1 | ISL 8장(트리) → scikit-learn 앙상블 가이드 | 반나절 | 모델 A의 기초 |
| 2 | Breiman 2001 (§1~3만) | 1~2시간 | RF가 왜 작동하는지의 원리 |
| 3 | scikit-learn permutation_importance 문서 | 30분 | 5.1.1절 분석 준비, MDI 편향 이해 |
| 4 | Bergstra & Bengio 2012 (랜덤서치) | 1시간 | BO 이해의 전 단계 |
| 5 | **Frazier 2018 튜토리얼 §1~3** | 2~3시간 | 모델 B의 핵심 |
| 6 | Bergstra et al. 2011 (TPE) §4 | 1시간 | 우리가 실제로 쓸 알고리즘 |
| 7 | scikit-learn StackingClassifier 문서 | 30분 | 모델 C, **OOF 개념 반드시 확인** |
| 8 | ISL 10장(딥러닝 기초) | 반나절 | 모델 D |

**시간이 부족하면**: 3번(순열 중요도)과 7번(OOF)은 건너뛰지 말 것. 이 둘이 우리 실험에서 실수가 나기 가장 쉬운 지점이다.

---

## 7. 실험 진행 시 자주 틀리는 지점 체크리스트

| 지점 | 흔한 실수 | 프로토콜 조항 |
|---|---|---|
| 스태킹 메타모델 학습 | in-sample 예측 사용 | 4.2절 (OOF 5-fold) |
| BO 목적함수 | Test 점수로 최적화 | 2.3절, 4.2.1절 |
| MLP 스케일링 | 전체 데이터로 fit | 2.3절, 3.4절 |
| 특징 중요도 | MDI만 보고 판단 | 5.1.1절 (순열 중요도 병행) |
| RF 깊이 설정 | 무제한으로 두고 프라이버시 미고려 | 10.1절 (정규화 = 프라이버시 개선안) |
| 모델 비교 | 개별 신뢰구간 중첩으로 유의성 판정 | 8장 (차이의 CI 사용) |
