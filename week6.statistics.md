# 통계학 6주차 정규과제

📌통계학 정규과제는 매주 정해진 분량의 『*데이터 분석가가 반드시 알아야 할 모든 것*』 을 읽고 학습하는 것입니다. 이번 주는 아래의 **Statistics_6th_TIL**에 나열된 분량을 읽고 `학습 목표`에 맞게 공부하시면 됩니다.

아래의 문제를 풀어보며 학습 내용을 점검하세요. 문제를 해결하는 과정에서 개념을 스스로 정리하고, 필요한 경우 추가자료와 교재를 다시 참고하여 보완하는 것이 좋습니다.

6주차는 `3부. 데이터 분석하기`를 읽고 새롭게 배운 내용을 정리해주시면 됩니다.

## Statistics_6th_TIL

### 3부. 데이터 분석하기

### 12.통계 기반 분석 방법론

## Study Schedule

| 주차 | 공부 범위 | 완료 여부 |
| --- | --- | --- |
| 1주차 | 1부 p.2~56 | ✅ |
| 2주차 | 1부 p.57~79 | ✅ |
| 3주차 | 2부 p.82~120 | ✅ |
| 4주차 | 2부 p.121~202 | ✅ |
| 5주차 | 2부 p.203~254 | ✅ |
| 6주차 | 3부 p.300~356 | ✅ |
| 7주차 | 3부 p.357~615 | 🍽️ |

<!-- 여기까진 그대로 둬 주세요-->

# 12.통계 기반 분석 방법론

```
✅ 학습 목표 :
* 주성분 분석(PCA)의 개념을 설명할 수 있다.
* 다중공선성을 진단할 수 있다.
* Z-TEST와 T-TEST의 개념을 비교하고, 적절한 상황에서 검정을 설계하고 수행할 수 있다.
* ANOVA TEST를 활용하여 세 개 이상의 그룹 간 평균 차이를 검정하고, 사후검정을 수행할 수 있다.
* 카이제곱 검정을 통해 범주형 변수 간의 독립성과 연관성을 분석하는 방법을 설명할 수 있다.

```

## 12.1. 분석 모델 개요

### 🧠방법론

- 통계 모델: 통계학에 기반
- 기계학습: 인공지능에서 파생

### 🧠 두 방법론의 차이

- 통계 모델 : 모형과 해석을 중요하게 생각하며, 오차와 불확실성을 강조
- 기계 학습 : 대용량 데이터를 활용하여 예측의 정확도를 높이는 것을 중요하게 생각
- 둘 다 같이 활용할 때 높은 성과를 얻을 수 있다.

![image](https://github.com/user-attachments/assets/4b7dbafc-cb56-4329-9890-a6d1866b1d83)

![image](https://github.com/user-attachments/assets/a4836bd0-4342-48f0-9d9a-200585a573fc)

![image](https://github.com/user-attachments/assets/1c307622-8e43-4113-9236-b20e69a31f77)


### 🧠 기계학습 데이터 분석 방법론

- **종속 변수의 유무에 따라**
    - 지도 학습
    - 비지도 학습
    - 강화학습
- **독립변수와 종속 변수의 속성에 따라**
    - 질적 척도
    - 양적 척도
    - 둘 다 가능한 것도 있다
        - 회귀모델은 독립 변수가 질적 양적 변수인 경우 모두 사용 가능
        - k-근접 이웃 모델은 종속 변수가 질적, 양적 모두 사용 가능

![image](https://github.com/user-attachments/assets/ccd61c51-da13-48cd-8dd0-eddc7bf96a55)


### **🕊 지도 학습**

- 입력에 대한 정답이 주어져서 출력된 결괏값과 정답 사이의 오차가 줄어들도록 학습과 모델 수정을 반복
- 결괏값이 양적 척도 → **`회귀`**
- 결괏값이 질적 척도 → **`분류`**

### **💸 비지도 학습(자율 학습)**

- 별도의 정답이 없어 변수 간의 패턴을 파악하거나 데이터의 군집화
- 차원 축소
    - 지도 학습할 때 학습 성능을 높이기 위한 **전처리 방법으로 사용되는 경우 多**
    - 통계 모델과 유사한 성격을 가짐
- 군집 분석
    - 정답지 없이 유사한 관측치들끼리 군집으로 분류하는 기법
    - 유사한 특성을 가진 것들끼리 묶으면서 각 집단의 특성을 분석
    - ex. 고객 세분화

### 🎶연관 규칙(장바구니 분석)

- 제품이나 콘텐츠를 추천하기 위해 사용하는 모델
- 각 소비자의 구매 리스트를 통해 제품 간의 연관성을 수치화하며 **소비자가 앞으로 구매할 가능성이 높은 제품 추천**

### ⛄ 강화학습

- 동물이 시행착오를 통해 학습하는 과정을 기본 콘셉트로 한 방법 중 하나
- 모델에게 보상과 벌을 주면서 스스로 학습하게 하는 것
- 이세돌 vs 알파고

## 12.2. 주성분 분석(PCA)

### 🐴 주성분 분석

- 여러 개의 독립 변수들을 잘 설명해 줄 수 있는 주된 성분을 추출하는 기법
- 전체 변수들의 핵심 특성만 선별하므로 독립변수(차원)의 수를 줄인다
- 차원의 저주를 방지
    - *데이터학습을 위해 차원이 증가하면서 학습 데이터 수가 차원의 수보다 적어져 성능이 저하되는 현상*
    - *차원이 증가할 수록 개별 차원 내 학습할 데이터 수가 적어지는(sparse) 현상 발생*
- 변수의 수를 줄이면서 모형을 간단하게 만들고 분석 결과를 보다 효과적으로 해석

### 🤹‍♂️주성분 분석의 조건

- 사용되는 변수들이 모두 등간 척도나 비율 척도로 측정한 양적 변수
- 관측치들이 서로 독립적이고 정규분포를 이루고 있어야 함

### 🏋️‍♀️차원을 감소하는 방법

1. 변수 선택을 통해 비교적 불필요하거나 유의성이 낮은 변수를 제거하는 방법
2. 변수들의 잠재적인 성분을 추출하여 차원을 줄이는 방법
    - ex. 주성분 분석(PCA), 공통요인분석(CFA)
    - 다차원의 데이터를 가장 설명해주는 성분을 찾아서, **분산을 최대한 보존하는 축을 통해** 차원을 축소
    - 처음 가지고 있던 변수의 개수만큼의 새로운 성분 변수가 생성
    - but, 전체 변수를 통합적으로 가장 잘 설명해주는 성분 변수, 그다음으로 높은 설명 변수… 이런식으로 주성분 변수가 생성
    - ⇒ 전체 변수 설명력이 높은 주성분 변수만 선정하여 총 변수의 개수를 줄일 수 있다.
    - 일반적으로 *제1주성분, 제2주성분만으로 대부분의 설명력이 포함되므로* 두 개의 주성분만 선정

### 👶 예시

![image](https://github.com/user-attachments/assets/27f485c2-d4b0-42fd-8f95-3e7b881a4c25)


- 1번,2번,3번 중에서 3번 축이 가장 많은 분산을 담아낼 수 있다 ⇒ 해당 차워의 가장 많은 분산을 담아내는 축 = 주성분
- 축과 각 포인트와의 거리를 나타내는 직각의 선을 보면,
    
    ![image.png](attachment:c2ac0659-d965-4035-9940-487b1f8962da:image.png)
    
    - 직각으로 맞닿는 지점의 분포가 가장 넓게 퍼진 축을 구하는 것
    - 각 포인트들의 직각 지점까지의 거리의 합이 가장 큰 축의 주성분!
    - 피타고라스의 정리에 의해 각 포인트와 주성분 축과의 거리의 합은 **최소가 됨**
- 가장 설명력이 높았던 3번 축과 직교하는 선 = 제2주성분
    
   ![image](https://github.com/user-attachments/assets/2988053f-d7c2-434e-8fdd-1f9ceb2f854c)

    
    - 3번 축과 달리 짧은 길이, 즉 낮은 분산
    - 제2주성분은 전체 변수에 대한 설명력이 낮다
    - 예시는 2차원이므로 최대 2개의 주성분을 만들 수 있다

### 🚣‍♂️각 주성분의 설명력은 어떻게 알 수 있을까?

- 전체 분산 중에서 해당 주성분이 갖고 있는 분산이 곧 설명력
- 각 주성분의 분산은 모든 포인트들과 주성분과의 걸의 제곱합/n-1

![image](https://github.com/user-attachments/assets/bad17128-a1ae-40d9-b2ef-5dff71b487c9)


- if: 제1주성분의 분산이 15, 제2주성분의 분산이 5라고 할 때,
    - 제 1주성분의 설명력=15/20=75%
    - 제 2주성분의 설명력=2/20=25%

### 🤸‍♀️주성분 분석 실습

```sql
df1 = df.drop('Type', axis=1)

MinMaxScaler = MinMaxScaler()
df_minmax = MinMaxScaler.fit_transform(df1)

df_minmax = pd.DataFrame(data=df_minmax, columns=df1.columns)

df_minmax.head()
```

```sql
pca = PCA(n_components=9) 
df_pca = pca.fit_transform(df_minmax)

df_pca = pd.DataFrame(data=df_pca, columns = ['C1','C2','C3','C4','C5','C6','C7','C8','C9'])

np.round_(pca.explained_variance_ratio_,3)
```

```sql
pca = PCA(n_components=2) 
df_pca = pca.fit_transform(df_minmax)

df_pca = pd.DataFrame(data=df_pca, columns = ['C1','C2'])

df_pca.head()
```

```sql
df_concat = pd.concat([df_pca,df[['Type']]],axis=1)

# 산점도 시각화
sns.scatterplot(data=df_concat,x='C1',y='C2',hue='Type')
```

![image](https://github.com/user-attachments/assets/1f75a922-5ffe-4d2f-a649-353825585ff4)


```sql
import matplotlib.pyplot as plt
plt.scatter(x=df_concat['C1'],y=df_concat['C2'],c=df_concat['Type'])
```

![image](https://github.com/user-attachments/assets/22bbba86-6d6e-4476-bc65-2ccf14095e20)

## 12.4. 다중공선성 해결과 섀플리 밸류 분석

### 🏄‍♀️ 다중공선성(multicollinerarity)

- 독립변수들 간의 상관관계가 높은 현상
- 두 개 이상의 독립변수가 서로 선형적인 관계 ⇒ 다중공선성이 있다!
- **독립 변수들 간 서로 독립**이라는 회귀분석의 가정 위반
- 첫번째 독립 변수가 종속변수를 예측하고 두번째 독립변수가 남은 변량을 예측하는 방식

![image](https://github.com/user-attachments/assets/989314a0-ff05-47b6-81e1-8fe509ff84ef)


### 🛌 다중공선성을 판별하는 기준

1. 회귀 진행 전에 상관분석을 통해 독립 변수 간의 상관성 확인 → 높은 상관 계수를 갖는 독립변수를 찾아냄
2. |  상관 계수 |≥ 0.7 ⇒ 두 변수간의 상관성이 높다 ⇒ 다중공선성 의심
    - 이 방법은 변수가 많을 경우 상관성을 파악하기 힘들다
3. 결정 계수 R^2값은 크지만, 회귀계수에 대한 t값(t-value)이 낮다 ⇒ 다중공선성 의심
4. 종속 변수에 대한 독립 변수들의 설명력은 높지만 각 계수 추정치의 표준오차가 큰 건 독립변수 간 상관성이 크다는 것을 의미
    - t값: 해당 변수의 시그널 강도
    - 표준 오차(노이즈) 대비 시그널이므로 값이 클수록 좋다
5. 관측치가 100개일 때, 95%의 신뢰도에서 t값이 +-1.96이상 ⇒ 적절
6. VIF(분산 팽창 지수)
    1. 해당 변수가 다른 변수에 의해 설명될 수 있는 정도
    2. VIF가 크다는 것은 ***해당 변수가 다른 변수들과 상관성이 높다***는 것을 의미, 회귀 계수에 대한 분산을 증가시킴(다른 변수들에 의해 설명되는 수준이 높을수록 VIF는 큰 값을 가진다.)
        
       ![image](https://github.com/user-attachments/assets/e529d4ad-df3c-4a79-9795-304e4a3225f2)

        
    3. VIF ≥ 5, 다중공선성 의심
    4. VIF ≥ 10, 다중공선성이 있다고 판단
    5. ex. A라는 변수의 VIF값=16이상, A변수는 다중공선성이 없는 상태보다 표준오차가 4배 높다

```
**🌌
다중 공선성을 해결하기 위한 가장 기본적인 방법:**
VIF값이 높은 변수들 중에서 
종속 변수와의 상관성이 가장 낮은 변수를 제거하고 다시 VIF값을 확인하는 것을 반복
```

### 🚀 표준 관측치를 추가적으로 확보 → 다중공선성을 완화

- 해당 값에 로그를 취하기
- 표준화 및 정규화 변환

→ 높은 상관성 완화 가능

- 연속형 변수를 구간화 혹은 명목 변수로 변환

*순수 변수를 가공하는 것이라 정보의 손실이 발생 but 다중 공선성 때문에 변수가 제거되는 것보다 나음

### 🏍 주성분 분석을 통한 다중공선성 제거

- 기존 변수의 변동을 가장 잘 설명하는 변수이므로 유사한 변수들을 하나의 변수로 합쳐낸 효과
    - 그래서, 변수의 해석이 어려움
- 변수 선택 알고리즘!
    - 전진 선택법
    - 후진 제거법
    - 단계적 선택법

### 🏖 섀플리 밸류(shapley value)

- 각 독립변수가 종속변수의 설명력에 기여하는 순수한 수치를 계산
    
    ![image](https://github.com/user-attachments/assets/3c0bcbdf-932f-4130-937f-543790f60d3a)

    
- x1 단일 조합에서 x1의 기여도: 0.15
- x1, x2와 x1,x3 두 변수 조합에서 x1의 기여도 평균: 0.115
    
    → (0.11+0.12)/2=0.115
    
- x1, x2, x3 조합에서 x1의 기여도: 0.04
- 모든 조합에서 x1의 기여도 평균: 0.101(x1의 섀플리 밸류)
    
    → (0.15+0.115+0.04)/3=0.101
    

## 12.6. Z-test와 T-test

- 집단 내 혹은 집단 간의 평균값 차이가 통계적으로 유의미한 것인지 알아내는 방법
- ex. 쇼핑몰의 지역별 객단가 분석
    - A지역의 고객별 매출: 67,000원
    - B지역의 고객별 매출: 68,500원
        - 1500원이 우연적인 차이인지
        - 통계적으로 유의미한 차이인지
    - **`z-test`**나 **`ANOVA`** 사용

### 🍧 z-test와 t-test

- 사용하는 경우
    - 단일 표본 집단의 평균 변화를 분석
    - 두 집단의 평균값 혹은 비율 차이를 분석
- 조건
    - 양적 변수
    - 정규 분포
    - 등분산
        - 등분산일 경우: equal variance t-test 수행
        - 이분산일 경우: welch’s t-test 수행
- 둘 중 선택하는 기준
    - 모집단의 분산을 알 수 있다면 z-test (거의 없음 ㅠㅠ)
        - n=30이상이면 중심 극한 정리에 의해 표본 평균 분포가 정규분포를 따른다고 할 수 있으므로 z-test 사용 가능
    - n=30이하, 모집단의 분산을 알 수 없을 때 t-test 사용
        - 알 수 있을 때도 사용할 수 있으므로 일반적으로 t-test 사용

![image](https://github.com/user-attachments/assets/849e483c-8016-4c3e-8701-d08b26da461e)


### 🥏T-test example

```
A 쇼핑몰 기존 고객집단 C의 월평균 매출은 42000원이었다.
마케팅 부서에서 새로운 고객 관리 프로그램을 개발해서 고객 집단 C에 적용을 했다.
새로운 마케팅 프로그램을 적용한 후 월평균 매출이 43000원으로 늘어났다.
1000의 증가가 통계적으로 유의미한 차이인지 분석해보자
(고객집단 C는 100명이다.)
```

**(1) 가설 세우기**

H0: 고객 집단 C의 마케팅 프로그램 적용 전후의 월평균 매출은 동일하다.

H1: 고객 집단 C의 마케팅 프로그램 적용 전후의 월평균 매출은 차이가 있다.

⇒ 양측 검정

- p-value가 0.25를 초과하는지에 따라 귀무가설의 기각여부 달라짐
    - 작으면 월평균 매출이 동일하다는 귀무가설 기각
    - 월평균 매출이 통계적으로 다르다는 결론

**(2) 평균, 표준편차, 표준 오차 계산**

![image](https://github.com/user-attachments/assets/bceb4466-a93e-4162-b5df-ceda2ce6d674)

**(3) t통계량 계산**

![image](https://github.com/user-attachments/assets/92103dd6-83a8-4804-900e-b244b548f916)


- 평균의 차이가 클수록, 표본의 수가 클수록 t값 증가
- 관측치들의 값 간의 표준편차가 크면 평균의 차이가 불분명해지고 t값은 감소

⇒ 고객들의 마케팅 프로그램 전과 후의 매출 차이가 들쑥날쑥하면 평균의 차이가 우연에 의한 것일 확률이 높음

- t값: 1.6 ≤ 1.984 ⇒ 효과가 없다고 판단

**(4) z-test**

- 모집단의 분산을 알고 있다는 가정으로 계산
- t-test: `집단 값 차이의 표준 편차` 사용 → z-test: `모집단의 표준 편차` 사용

**(5) 두 집단의 평균 차이 검정**

- 두 집단 값을 산포도로 확인하여 평균 차이 유의성 판단
- t-test와 동일한 방법으로 가설을 설정, 각 집단의 평균, 분산, 표본 평균 차이, 표준 오차 등의 값으로 t값 구하여 귀무가설 기각 여부 판단

![image](https://github.com/user-attachments/assets/d62d15a8-5a27-4fd5-b9f6-7e44f25e313f)


두 집단의 t-test를 하는 공식:

![image](https://github.com/user-attachments/assets/2ae28ef7-bef1-463f-9363-f3e7ce103589)

- 두 집단의 평균 차이에 대한 귀무가설 추정

**(6) 단일 집단의 비율 차이에 대한 t-test와 두 집단의 비율 차이 검정을 위한 t-test**

- 비율의 차이를 표준오차로 나눠서 t값을 계산 → 유의수준으로 환산하여 대립가설 채택 여부를 판단

![image](https://github.com/user-attachments/assets/b91c1093-b45d-4491-920b-a53e6866c1f7)


- 단일 집단 비율 검정

![image](https://github.com/user-attachments/assets/433c4c06-63b2-4fca-b09f-171388bc1a61)

- 두 집단 비율 차이에 대한 t-test

### 👕 z-test와 t-test 실습

```python
df.describe()
```

```python
df2 =  pd.melt(df)
plt.figure(figsize=(12,6))
sns.boxplot(x='variable', y='value', data=df2)
plt.title('Golf ball test')
plt.show()
```

![image](https://github.com/user-attachments/assets/f7c07357-3434-4112-bfcc-88b1e9c7f52b)

```python
print(shapiro(df['TypeA_before']))
print(shapiro(df['TypeA_after']))
print(shapiro(df['TypeB_before']))
print(shapiro(df['TypeB_after']))
print(shapiro(df['TypeC_before']))
print(shapiro(df['TypeC_after']))
```

```python
stats.bartlett(df['TypeA_before'],df['TypeA_after'],
               df['TypeB_before'],df['TypeB_after'],
               df['TypeC_before'],df['TypeC_after'])
```

```python
ztest(df['TypeA_before'], x2=df['TypeA_after'], value=0, alternative='two-sided')
```

```python
# 양측검정
print(ztest(df['TypeA_before'], x2=df['TypeB_before'], value=0, alternative='two-sided'))

# 단측검정(왼꼬리검정)
print(ztest(df['TypeA_before'], x2=df['TypeB_before'], value=0, alternative='smaller'))

# 단측검정(오른꼬리검정)
print(ztest(df['TypeA_before'], x2=df['TypeB_before'], value=0, alternative='larger'))
```

```python
scipy.stats.ttest_rel(df['TypeA_before'],df['TypeA_after'])
```

```python
ttest_ind(df['TypeA_before'],df['TypeB_before'], equal_var=False)
```

## 12.7. ANOVA

### 🎃 ANOVA

- 3집단 이상의 평균 차이를 검정할 때 사용
- t-test를 사용해서 할 수도 있지만 여러분 중복하여 검정하면 신뢰도가 하락함
- 분산 분석이라고도 함
- F-분포 사용

![image](https://github.com/user-attachments/assets/cbe90031-87c3-4f76-8781-ad4eb5f43dce)


- 항상 양의 값을 가지며 오른쪽으로 긴 꼬리 형태
- 집단의 종류(독립 변수)가 평균 값의 차이 여부(종속 변수)에 미치는 영향을 검정

```
**[가설]**
H0: 독립변수(인자)의 차이에 따른 종속변수(특성 값)는 동일하다.
H1: 독립변수(인자)의 차이에 따른 종속변수(특성 값)는 다르다.
```

- 독립변수인 요인의 수에 따라서 다르게 불린다
    - 요인이 1개 : 일원 분산 분석(one-way ANOVA)
    - 요인이 2개: 이원 분산 분석(two-way ANOVA)
    - 요인이 n개: n원 분산분석(N-way ANOVA)
- 독립변수: 범주형 변수
- 종속변수: 연속형 변수
    - 독립변수와 종속변수에 따라 회귀분석이나 교차분석으로 분석 방법이 바뀐다
    - 회귀분석: 독립변수와 종속변수가 **`연속형`**일 때 사용
    - 교차분석: 독립변수와 종속변수가 **`분류형`**일 때 사용
- 집단의 평균값 차이가 통계적으로 유의한지 검증
- **`집단 평균의 분산`**이 큰 정도를 따짐 → 집단 간 평균이 다른지 판별

![image](https://github.com/user-attachments/assets/a137bedc-ed49-4570-a3f1-78e2e8a9d7c5)

- 각 집단의 관측치 값은 서로 겹칠 수 있다

![image](https://github.com/user-attachments/assets/1e2534db-83bb-4176-901f-64260a8a940d)

- 집단 간의 분산을 집단내 분산으로 나눠서 F값 구하기

![image](https://github.com/user-attachments/assets/f5e29ad9-ea0a-4f80-bbc6-a990736dacc0)

![image](https://github.com/user-attachments/assets/bce7c44f-a028-45a7-a378-fdeadb6039d9)

- ex.
    - F값=5.73≥ 5.14
        - 집단 간 평균에는 통계적으로 유의미한 차이가 있다
        - 이 결과만으로는 연령대의 평균이 모두 다른건지 일부만 다른지 모른다.
        - ex. 집단 간 차이가 있지만 30대나 40대 간에는 차이가 없을 수도 있다
        - 1종 오류를 방지하기 위해 사후 검증(post hoc)을 한다.
- 사후 검증
    - 독립 변수 수준 사이에서 평균의 차이를 알고자 할 때 쓰이는 기법
    - 각 집단의 수가 같을 때 사용하는 Turkey의 HSD검증
    - 집단의 수가 다를 때 사용하는 Scheffe 검증

### 🖼 ANOVA 실습

```python
F_statistic, pVal = stats.f_oneway(df['TypeA_before'], 
                                   df['TypeB_before'], 
                                   df['TypeC_before'])

print('일원분산분석 결과 : F={0:.1f}, p={1:.5f}'.format(F_statistic, pVal))
```

```python
df2 =  pd.melt(df)
df2 = df2[df2['variable'].isin(['TypeA_before', 'TypeB_before', 'TypeC_before'])]

df2.head()
```

```python
model = ols('value ~ C(variable)', df2).fit()
print(anova_lm(model))
```

```python
posthoc = pairwise_tukeyhsd(df2['value'], 
                            df2['variable'], 
                            alpha=0.05)
print(posthoc)
fig = posthoc.plot_simultaneous()
```

![image](https://github.com/user-attachments/assets/84ba4691-78ec-468e-bb21-1d4a5abd0d41)


## 12.8. 카이제곱 검정(교차분석)

### 🥁 카이제곱 검정

- 등간이나 비율척도로 이루어진 연속형 변수 간의 연관성을 측정
- 명목 혹은 서열척도와 같은 범주형 변수들 간의 연관성을 분석하기 위해 결합 분포를 활용
- 연령과 같은 비율척도 변수는 연령대와 같은 서열척도로 변환해서 사용
- 검정 통계량 카이 제곱을 통해 변수 간에 연관성이 없다는 귀무가설을 기각하는지 여부로 상관성 판단
- ex.
    - 두 개의 변수를 교차표로 표현
        
        ![image](https://github.com/user-attachments/assets/c74468f6-b7c2-47e2-9478-07282d8a319e)

    - 두 변수가 서로 독립적일 경우에 나타날 기대빈도를 구한다
        
       ![image](https://github.com/user-attachments/assets/17dfb99e-9efe-4ec3-9d0f-944c54e0cfdd)

        
    - 검정 통계량값을 산출하여 유의수준에 따라 결정된 임계치를 비교하여 가설을 검정
    - 가설 검정하기 위해 임계치를 비교

![image](https://github.com/user-attachments/assets/d0259c82-7ae5-4489-a868-8058dae5980e)


### ☎ 카이제곱 검정 실습

```python
df.groupby(['sex','smoke'])['smoke'].count()
```

```python
crosstab = pd.crosstab(df.sex, df.smoke)
crosstab
```

```python
%matplotlib inline
crosstab.plot(kind='bar', figsize=(10,5))
plt.grid()
```

![image.png](attachment:cd7a23c6-9249-42a4-a70f-a84fb3023017:image.png)

```python
chiresult = chi2_contingency(crosstab, correction=False)
print('Chi square: {}'.format(chiresult[0]))
print('P-value: {}'.format(chiresult[1]))
```

# 확인 문제

### **문제 1.**

> 🧚 경희는 다트비 교육 연구소의 연구원이다. 경희는 이번에 새롭게 개발한 교육 프로그램이 기존 프로그램보다 학습 성취도 향상에 효과적인지 검증하고자 100명의 학생을 무작위로 두 그룹으로 나누어 한 그룹(A)은 새로운 교육 프로그램을, 다른 그룹(B)은 기존 교육 프로그램을 수강하도록 하였다. 실험을 시작하기 전, 두 그룹(A, B)의 초기 시험 점수 평균을 비교한 결과, 유의미한 차이가 없었다. 8주 후, 학생들의 최종 시험 점수를 수집하여 두 그룹 간 평균 점수를 비교하려고 한다.
> 

> 🔍 Q1. 이 실험에서 사용할 적절한 검정 방법은 무엇인가요?
> 

```
두 집단의 평균값 차이 분석이므로 t-test검정이 필요하다
```

> 🔍 Q2. 이 실험에서 설정해야 할 귀무가설과 대립가설을 각각 작성하세요.
> 

```
H0: 두 그룹(A, B)의 최종 시험 점수 평균의 차이가 없다.
H1: 두 그룹(A, B)의 최종 시험 점수 평균의 차이가 있다.
```

> 🔍 Q3. 검정을 수행하기 위한 절차를 순서대로 서술하세요.
> 

<!--P.337의 실습 코드 흐름을 확인하여 데이터를 불러온 후부터 어떤 절차로 검정을 수행해야 하는지 고민해보세요.-->

```
1) 각 몇 개의 관측치가 있는지 확인
	- 30개를 기준으로 z-test랑 t-test를 사용할 수 있는지 
2) 각 그룹의 평균값 확인
	- 각 그룹의 평균값을 확인했을 때 차이가 나는지 직관적으로 확인
3) 박스 플롯 시각화
	- 평균 값을 직관적으로 확인하기 위해 시각화 진행
4) 데이터 정규성 검정
	- 모수적 방법을 쓸 수 있는지 체크하기 위해서 정규성을 확인
5) 등분산성 검정
	- 등분산일 경우: equal variance t-test 수행
	- 이분산일 경우: welch’s t-test 수행
6) t-test 검정(양측 검정)
```

> 🔍 Q4. 이 검정을 수행할 때 가정해야 하는 통계적 조건을 설명하세요.
> 

```
양적변수이며 정규분포, 등분산이여야 한다. 
```

> 🔍 Q5. 추가적으로 최신 AI 기반 교육 프로그램(C)도 도입하여 기존 프로그램(B) 및 새로운 프로그램(A)과 비교하여 성취도 차이가 있는지 평가하고자 한다면 어떤 검정 방법을 사용해야 하나요? 단, 실험을 시작하기 전, C 그룹의 초기 점수 평균도 A, B 그룹과 유의미한 차이가 없었다고 가정한다.
> 

```
3개의 집단의 평균을 비교하는 것이기 때문에 ANOVA를 사용해야 한다.
```

> 🔍 Q6. 5번에서 답한 검정을 수행한 결과, 유의미한 차이가 나타났다면 추가적으로 어떤 검정을 수행해 볼 수 있을까요?
> 

```
1종 오류를 방지하기 위해 사후 검증(post hoc)을 한다.
독립 변수 수준 사이에서 평균의 차이를 알고자 할 때 쓰이는 기법으로
각 집단의 수가 같을 때 사용하는 Turkey의 HSD검증, 
집단의 수가 다를 때 사용하는 Scheffe 검증을 한다.
```

---

### **문제 2. 카이제곱 검정**

> 🧚 다음 중 어떠한 경우에 카이제곱 검정을 사용해야 하나요?
1️⃣ 제품 A, B, C의 평균 매출 차이를 비교하고자 한다.
2️⃣ 남성과 여성의 신체 건강 점수 평균 차이를 분석한다.
3️⃣ 제품 구매 여부(구매/미구매)와 고객의 연령대(10대, 20대, 30대…) 간의 연관성을 분석한다.
4️⃣ 특정 치료법이 환자의 혈압을 감소시키는 효과가 있는지 확인한다.
> 

```
3. 제품 구매 여부(구매/미구매)와 고객의 연령대(10대, 20대, 30대…) 간의 연관성을 분석한다.
```

### 🎉 수고하셨습니다.
