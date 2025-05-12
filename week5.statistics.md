# 통계학 5주차 정규과제

📌통계학 정규과제는 매주 정해진 분량의 『*데이터 분석가가 반드시 알아야 할 모든 것*』 을 읽고 학습하는 것입니다. 이번 주는 아래의 **Statistics_5th_TIL**에 나열된 분량을 읽고 `학습 목표`에 맞게 공부하시면 됩니다.

아래의 문제를 풀어보며 학습 내용을 점검하세요. 문제를 해결하는 과정에서 개념을 스스로 정리하고, 필요한 경우 추가자료와 교재를 다시 참고하여 보완하는 것이 좋습니다.

5주차는 `2부. 데이터 분석 준비하기`를 읽고 새롭게 배운 내용을 정리해주시면 됩니다.

## Statistics_5th_TIL

### 2부. 데이터 분석 준비하기

### 11.데이터 전처리와 파생변수 생성

## Study Schedule

| 주차 | 공부 범위 | 완료 여부 |
| --- | --- | --- |
| 1주차 | 1부 p.2~56 | ✅ |
| 2주차 | 1부 p.57~79 | ✅ |
| 3주차 | 2부 p.82~120 | ✅ |
| 4주차 | 2부 p.121~202 | ✅ |
| 5주차 | 2부 p.203~254 | ✅ |
| 6주차 | 3부 p.300~356 | 🍽️ |
| 7주차 | 3부 p.357~615 | 🍽️ |

<!-- 여기까진 그대로 둬 주세요-->

# 11.데이터 전처리와 파생변수 생성

```
✅ 학습 목표 :
* 결측값과 이상치를 식별하고 적절한 방법으로 처리할 수 있다.
* 데이터 변환과 가공 기법을 학습하고 활용할 수 있다.
* 모델 성능 향상을 위한 파생 변수를 생성하고 활용할 수 있다.

```

## 11.1. 결측값 처리

- 가공되지 않은 데이터는 상당량의 결함을 가지고 있음

> **처리 방법을 결정하기 전 확인할 것**
> 
- 결측값의 비율이 어떻게 되는지,
- 한 변수에 결측값이 몰려있지 않은지,
- 빈 문자열이 입력되어 있어 결측값으로 인식되지 않는지

> **결측값이 발생하는 특성**
> 
- 완전 무작위 결측
    - 순수하게 결측값이 무작위로 발생한 경우
- 무작위 결측
    - 다른 변수의 특성에 의해 해당 변수의 결측치가 체계적으로 발생한 경우
    - 특정 체인점의 pos기기에 오류가 나서 해당 체인점에 해당하는 매출 정보에 결측값이 많이 난 경우
- 비무작위 결측
    - 결측값들이 해당 변수 자체의 특성을 갖고 있는 경우
    - 고객 소득 변수에서 결측값 대부분이 소득이 적어서 소득을 공개하기 꺼려서 결측이 발생한 경우
    - 결측된 값은 그 값이 무엇인지 확인할 수 없으므로 비무작위 결측을 구분하는 것은 어렵다

> **결측값 처리 방법**
> 
- 표본 제거 방법
    - 결측값이 심하게 많은 변수를 제거하거나 결측값이 포함된 행을 제외하고 데이터 분석
    - 결측값 비율이 10% 비만일 경우 많이 사용

생각보다 높은 비율의 결측값이 있는 경우, 연속형 변수에서 결측치를 처리하는 경우가 많기 때문에 이를 고려하면,

- 평균 대치법
    - 결측값을 제외한 온전한 값들의 평균을 구한 다음, 그 평균 값을 결측값들에 대치
    - 최빈값, 중앙값, 최댓값, 최솟값 대치
    - 관측 데이터의 평균을 사용하기 때문에 통계량의 표본 오차가 왜곡되어 축소
    
    → p-value가 부정확하게 된다
    
- 보간법(interpolation)
    - 시계열적 특성을 가지고 있을 경우
    - 전 시점의 값, 다음 시점의 값으로 대치하거나 전 시점과 다음 시점의 평균값으로 대치
    - 단, 시점 인덱스값이 불규칙하거나, 결측값이 두 번 이상 연달아 있을 때는 **선형적인 수치 값을 계산해 보간함**
- 회귀 대치법
    - 해당 변수와 다른 변수 사이의 관계성을 고려하여 결측값을 계산하면 보다 합리적으로 결측값 처리 가능
    - 결측된 변수의 분산을 과소 추정함
- 확률적 회귀 대치법
    - 인위적으로 회귀식에 확률 오차항을 추가
    - 관측된 값들을 변동성만큼 결측값에도 같은 변동성을 추가
    - 어느정도 표본 오차를 과소 추정

→ 단순 대치법들의 표본오차 과소 추정 문제를 해결하기 위해 많이 쓰는 방법

- 다중 대치법
    - 단순대치를 여러 번 수행하여 n 개의 가상적 데이터를 생성하여 이들의 평균으로 결측값을 대치하는 방법
    1. 대치 단계: 가능한 대치 값의 분포에서 추출된 서로 다른 값으로 결측치를 처리한 n개의 데이터셋 생성
        - 몬테카를로 방법
        - 연쇄방정식을 통한 다중 대치 사용
        - 5개 내외 정도 생성
        - but 결측값의 비율이 증가할수록 가상 데이터도 많이 생성해야 검정력 증가
    2. 분석 단계: 생성된 각각의 데이터셋을 모수의 추정치와 표준오차 계산
    3. 결합 단계: 계산된 각 데이터셋의 추정치와 표준오차를 결합하여 최종 결측 대치값 산출

![image](https://github.com/user-attachments/assets/ca7ff1b9-49b1-4038-9579-9c294beb2f1e)

## 11.1.1 결측값 처리 실습

```python
df=pd.read_csv("bike_sharing_daily.csv")
df.info()
#결측값 수 확인
df.isnull().sum()
```

```python
#결측값 영역 표시
msno.matrix(df)
plt.show()

#결측값 막대 그래프
msno.bar(df)
plt.show()
```

![image](https://github.com/user-attachments/assets/e604afed-679c-4703-a154-275dba0b8399)


```python
#결측값이 아닌 빈 문자열이 있는지 확인
def is_emptystring(x):
	return x.eq('').any()

df.apply(lambda x:is_emptystring(x))

```

```python
#모든 칼람이 결측값이 행 제거
df_drop_all=df.dropna(how='all')

#세 개 이상의 칼럼이 결측값인 행 제거
df_drop_3=dfdropna(thresth=3)

#특정 칼럼이 결측값인 행 제거
df_drop_slt=df.dropna(subset=['temp'])

#한 칼럼이라도 결측치가 있는 행 제거
df_drop_any=df.dropna(how='any')
df_drop_any.isnull().sum()

```

```python
## 결측값 기본 대치 방법들

# 특정값(0)으로 대치 - 전체 컬럼
df_0_all = df.fillna(0)

# 특정값(0)으로 대치 - 컬럼 지정
df_0_slt = df.fillna({'temp':0})

# 평균값 대치 - 전체 컬럼
df_mean_all = df.fillna(df.mean())

# 평균값 대치 - 컬럼 지정
df_mean_slt = df.fillna({'temp':df['temp'].mean()})

# 중앙값 대치 - 전체 컬럼
df_median_all = df.fillna(df.median())

# 중앙값 대치 - 컬럼 지정
df_median_slt = df.fillna({'temp':df['temp'].median()})

# 최빈값 대치 - 전체 컬럼
df_mode_all = df.fillna(df.mode())

# 최빈값 대치 - 컬럼 지정
df_mode_slt = df.fillna({'temp':df['temp'].mode()})

# 최댓값 대치 - 전체 컬럼
df_max_all = df.fillna(df.max())

# 최댓값 대치 - 컬럼 지정
df_max_slt = df.fillna({'temp':df['temp'].max()})

# 최솟값 대치 - 전체 컬럼
df_min_all = df.fillna(df.min())

# 최솟값 대치 - 컬럼 지정
df_min_slt = df.fillna({'temp':df['temp'],'hum':df['hum'].min()})

df_min_slt.isnull().sum()
```

```python
# 결측값 보간 대치 방법들

# 전 시점 값으로 대치 - 컬럼 지정
df1 = df.copy()
df1['temp'].fillna(method ='pad' ,inplace=True)

# 뒤 시점 값으로 대치 - 전체 컬럼
df.fillna(method ='bfill')

# 뒤 시점 값으로 대치 - 결측값 연속 한번만 대치
df.fillna(method='bfill', limit=1)

# 보간법 함수 사용하여 대치 - 단순 순서 방식
ts_intp_linear = df.interpolate(method='values')

# 보간법 함수 사용하여 대치 - 시점 인덱스 사용

    # dteday 컬럼 시계열 객체 변환
df['dteday'] = pd.to_datetime(df['dteday'])

    # dteday 컬럼 인덱스 변경
df_i = df.set_index('dteday') 

    # 시점에 따른 보간법 적용
df_time = df_i.interpolate(method='time')

df_time.isnull().sum()
```

```python
#다중 대치(MICE) 

# dteday 컬럼 제거
df_dp = df.drop(['dteday'],axis=1)

# 다중 대치 알고리즘 설정
imputer=IterativeImputer(imputation_order='ascending',
                         max_iter=10,random_state=42,
                         n_nearest_features=5)

# 다중 대치 적용
df_imputed = imputer.fit_transform(df_dp)

# 판다스 변환 및 컬럼 설정
df_imputed = pd.DataFrame(df_imputed)
df_imputed.columns = ['instant','season','yr','mnth','holiday'
                    ,'weekday','workingday','weathersit','temp'
                    ,'atemp','hum','windspeed','casual','registered','cnt']

df_imputed.isnull().sum()
```

## 11.2. 이상치 처리

### 이상치란,

**일부 관측치의 값이 전체 데이터의 범위에서 크게 벗어난 아주 작거나 큰 극단적인 값을 갖는 것**

- 전체 데이터의 양이 많을수록 튀는 값이 통곗값에 미치는 영향력이 줄어들어 이상치 제거의 필요성이 낮아진다.

![image](https://github.com/user-attachments/assets/bee1b367-2217-4df3-bfcd-f7dbb0b09e5a)

- 좌측 상단에 이상치 하나가 포함됨으로 인해 회귀선의 경사가 왜곡 → 극단적인 값은 데이터 분석 모델의 예측력을 약화시키는 주요 원인

**⇒ 해당 이상치를 제거(trimming)**

but, 추정치의 분산은 감소하지만 실젯값은 과장 → 편향 발생

**해결방법**

- 관측값 변경
    - 하한 값과 상한 값을 결정한 후
    - 하한 값보다 작으면 하한 값으로 대체, 상한 값보다 크면 상한 값으로 대체
    
    ![image](https://github.com/user-attachments/assets/a3378b79-11d5-4803-a48d-71ea098587a9)

    
- 가중치 조정
    - 이상치의 영향을 감소시키는 가중치

**이상치 선정 방법**

- 박스플롯 상에서 분류된 극단치를 그대로 선정
- 임의로 허용범위를 설정하여 이를 벗어나는 자료를 이상치로 정의
- 평균으로부터 +-n 표준편차 이상 떨어져 있는 값 = 이상치
    - 보통 3으로 하지만 경우에 따라 다르다
- 평균은 이상치 통계량에 민감하기 때문에 **이상치보다 강한 중위수와 중위수 절대편차를 사용하는 것이 더 효과적**

```
**주의**
통계치를 통한 무조건적인 이상치 탐색은 위험!
분석 도메인에 다라 이상치가 중요한 분석 요인일 수 있다
```

## 11.2.1 이상치 처리 실습

```python
plt.figure(figsize = (8, 6))
sns.boxplot(y = 'BMI', data = df)
plt.show()
```

![image](https://github.com/user-attachments/assets/9e889d5f-27a3-4535-9372-759598e6652b)

```python
# BMI 컬럼의 이상치 제거 (IQR*3)

# Q!, Q3 범위 정의
Q1 = df['BMI'].quantile(0.25)
Q3 = df['BMI'].quantile(0.75)
IQR = Q3 - Q1    #IQR 범위. 
rev_range = 3  # 제거 범위 조절 변수 설정

# 이상치 범위 설정
filter = (df['BMI'] >= Q1 - rev_range * IQR) & (df['BMI'] <= Q3 + rev_range *IQR)
df_rmv = df.loc[filter]
print(df['BMI'].describe())
print(df_rmv['BMI'].describe())
```

```python
# 이상치 제거 후 박스플롯 시각화

plt.figure(figsize = (8, 6))
sns.boxplot(y = 'BMI', data = df_rmv)
plt.show()
```

![image](https://github.com/user-attachments/assets/70a38c54-b97b-4ded-b7d3-7d5ebce3c7a1)

```python
# 이상치 IQR*3 값으로 대치

# 이상치 대치 함수 설정
def replace_outlier(value):
    Q1 = df['BMI'].quantile(0.25)
    Q3 = df['BMI'].quantile(0.75)
    IQR = Q3 - Q1    #IQR 범위. 
    rev_range = 3  # 제거 범위 조절 변수 설정

    if ((value < (Q1 - rev_range * IQR))): 
        value = Q1 - rev_range * IQR
    if ((value > (Q3 + rev_range * IQR))): 
        value = Q3 + rev_range * IQR
#         value = df['BMI'].median() # 중앙값 대치
    return value
df['BMI'] = df['BMI'].apply(replace_outlier)

print(df['BMI'].describe())
```

```python
# 이상치 대치 후 박스플롯 시각화

plt.figure(figsize = (8, 6))
sns.boxplot(y = 'BMI', data = df)
plt.show()
```

## 11.3. 변수 구간화(Binning)

- 목적
    - 데이터 분석의 성능을 향상시키기 위해,
    - 해석의 편리성을 위해
- 변수 구간화
    
    ![image](https://github.com/user-attachments/assets/82cbb4f6-2b23-48df-b6c6-8f4b36649de9)

    
    - 비즈니스적 상황에 맞도록 이산형 변수 ⇒ 범주형 변수 변환
    - 데이터의 해석이나 예측, 분류 모델을 의도에 맞도록 유도
- 평활화(smoothing)
    - 변수의 값을 일정한 폭이나 빈도로 구간을 나눔
    - 구간 안에 속한 데이터 값을 평균, 중앙값, 경곗값 등으로 변환
        - 머신 러닝 기법: 클러스터링, 의사결정 나무
- 효과적으로 구간화 되었는지 확인
    - WOE(weight of evidence)값
    - IV(information value)값
    
    ![image](https://github.com/user-attachments/assets/91827d57-83fb-41ef-9417-2939bb88ee2e)

    
    - 변수가 종속 변수를 제대로 설명할 수 있도록 구간화가 잘되면 IV값이 높아지는 것

### 11.3.1 변수 구간화 실습

```python
# BMI 컬럼의 분포 확인

df['BMI'].describe()
```

```python
# BMI 컬럼 분포 시각화

%matplotlib inline
sns.displot(df['BMI'],height = 5, aspect = 3)
```

![image](https://github.com/user-attachments/assets/bf5457d0-7d44-4bbd-9296-548f9d33e69a)


```python
# 임의로 단순 구간화

df1 = df.copy() # 데이터셋 복사

# 구간화용 빈 컬럼 생성 - 생략해도 되지만 바로 옆에 붙여 보기 위함
df1.insert(2, 'BMI_bin', 0) 

df1.loc[df1['BMI'] <= 20, 'BMI_bin'] = 'a'
df1.loc[(df1['BMI'] > 20) & (df1['BMI'] <= 30), 'BMI_bin'] = 'b'
df1.loc[(df1['BMI'] > 30) & (df1['BMI'] <= 40), 'BMI_bin'] = 'c'
df1.loc[(df1['BMI'] > 40) & (df1['BMI'] <= 50), 'BMI_bin'] = 'd'
df1.loc[(df1['BMI'] > 50) & (df1['BMI'] <= 60), 'BMI_bin'] = 'e'
df1.loc[(df1['BMI'] > 60) & (df1['BMI'] <= 70), 'BMI_bin'] = 'f'
df1.loc[df1['BMI'] > 70, 'BMI_bin'] = 'g'

df1.head()
```

```python
# 구간화 변수 분포 시각화

sns.displot(df1['BMI_bin'],height = 5, aspect = 3)
```

![image](https://github.com/user-attachments/assets/f65d00f4-5a84-422d-a794-73db78d51469)

```python
#  cut() 함수 사용하여 임의로 구간화
df1.insert(3, 'BMI_bin2', 0) # 구간화용 빈 컬럼 생성 

df1['BMI_bin2'] = pd.cut(df1.BMI, bins=[0, 20, 30, 40, 50, 60, 70, 95]
                         , labels=['a', 'b', 'c', 'd', 'e', 'f', 'g'])

df1.head()
```

```python
# BMI_bin2 구간 별 관측치 수 집계

df1.BMI_bin2.value_counts().to_frame().style.background_gradient(cmap='winter')
```

```python
# qcut() 함수 사용하여 자동 구간화
df1.insert(4, 'BMI_bin3', 0) # 구간화용 빈 컬럼 생성 

df1['BMI_bin3'] = pd.qcut(df1.BMI, q=7, labels=['a', 'b', 'c', 'd', 'e', 'f', 'g'])

df1.head()
```

```python
# BMI_bin3 구간 별 관측치 수 집계

df1.BMI_bin3.value_counts().to_frame().style.background_gradient(cmap='winter')
```

```python
# BMI_bin3 분포 시각화

sns.displot(df1['BMI_bin3'],height = 5, aspect = 3)
```

![image](https://github.com/user-attachments/assets/ef6e3d41-4861-46fc-9af0-7793f5b57ca0)


```python
# WOE를 사용한 변수 구간화

df2 = df.copy()  # 데이터셋 복사

# xverse 함수 적용을 위한 더미변수 변환
df2=pd.get_dummies(df)

# 구간화 할 컬럼(X), 기준 컬럼(y) 지정
X = df2[['PhysicalHealth']]
y = df2[['KidneyDisease_Yes']]

y = y.T.squeeze() # 차원 축소

# WOE 모델 설정 및 적용
clf = WOE()
clf.fit(X, y)

# 구간 기준점 및 eight of Evidence 값 테이블 생성
a=clf.woe_df 

#Information Value 데이블 생성
b=clf.iv_df

a.head()
```

```python
# Information Value 확인

b.head()
```

## 11.4. 데이터 표준화와 정규화 스케일링

- 표준화
    - 각 관측치의 값이 전체 평균을 기준으로 어느 정도 떨어져 있는지 나타낼 때 사용
- 정규화
    - 데이터의 범위를 0부터 1까지 변환하여 데이터 분포를 조정하는 방법

![image](https://github.com/user-attachments/assets/a2b93fe0-66f0-47b9-b458-5f5b33f75bb6)

![image](https://github.com/user-attachments/assets/41376c76-f9e3-45c6-aef2-979477f5463d)


- RobustScaler
    - 기본 표준화, 정규화 방식은 이상치에 민감하다는 단점을 보완한 스케일링 기법
    - 데이터 중앙값(Q2)을 0으로 잡고, Q1(25%)과 Q3(75%) 사분위수와의 IQR 차이를 1이 되도록 하는 스케일링 기법
    - 이상치의 영향력을 최소화하여 일반적으로 표준화 정규화보다 성능이 우수

### 11.4.1 데이터 표준화와 정규화 스케일링 실습

```python
# 각 컬럼의 평균값
print(df.mean())
print('\n')
# 각 컬럼의 분산값
print(df.var())
```

```python
# 데이터 표준화 적용
StandardScaler = StandardScaler()
df_stand = StandardScaler.fit_transform(df)

# 컬럼명 결합
df_stand = pd.DataFrame(data=df_stand, columns=df.columns)

df_stand.head()
```

```python
# 데이터 표준화 스케일링 후 컬럼 별 평균, 분산 확인

# 각 컬럼의 평균값
print(df_stand.mean())
print('\n')
# 각 컬럼의 분산값
print(df_stand.var())
```

```python
# Magnesium 컬럼 표준화 스케일링 전과 후 분포 비교

%matplotlib inline
sns.displot(df['Magnesium'],height = 5, aspect = 3)
sns.displot(df_stand['Magnesium'],height = 5, aspect = 3)

plt.show()
```

```python
# 전체 컬럼 정규화

# 데이터 정규화 적용
MinMaxScaler = MinMaxScaler()
df_minmax = MinMaxScaler.fit_transform(df)

# 컬럼명 결합
df_minmax = pd.DataFrame(data=df_minmax, columns=df.columns)

df_minmax.head()
```

```python
# 정규화 적용 컬럼 최솟값, 최댓값 확인

print(df_minmax.min()) #최솟값 
print('\n')
print(df_minmax.max()) #최댓값 
```

```python
# 데이터 정규화 스케일링 후 컬럼 별 평균, 분산 확인

# 각 컬럼의 평균값
print(df_minmax.mean())
print('\n')
# 각 컬럼의 분산값
print(df_minmax.var())
```

```python
# Magnesium 컬럼 정규화 스케일링 전과 후 분포 비교

%matplotlib inline
sns.displot(df['Magnesium'],height = 5, aspect = 3)
sns.displot(df_minmax['Magnesium'],height = 5, aspect = 3)

plt.show()
```

```python
# 전체 컬럼 RobustScaler

# 데이터 RobustScaler 적용
RobustScaler = RobustScaler()
df_robust = RobustScaler.fit_transform(df)

# 컬럼명 결합
df_robust = pd.DataFrame(data=df_robust, columns=df.columns)

df_robust.head()
```

```python
# 데이터 RobustScaler 적용 후 컬럼 별 평균, 분산 확인

# 각 컬럼의 평균값
print(df_robust.mean())
print('\n')
# 각 컬럼의 분산값
print(df_robust.var())
```

```python
# Magnesium 컬럼 RobustScaler 적용 전과 후 분포 비교

%matplotlib inline
sns.displot(df['Magnesium'],height = 5, aspect = 3)
sns.displot(df_robust['Magnesium'],height = 5, aspect = 3)

plt.show()
```

## 11.5. 모델 성능 향상을 위한 파생 변수 생성

- 파생변수
    
    ![image](https://github.com/user-attachments/assets/a4a7647e-462b-45f5-a3ff-646e0fd86e55)

    
    - 원래 있던 변수들을 조합하거나 함수를 적용하여 새로 만들어낸 변수
    - 데이터의 특성을 이용하여 분석 효율을 높이는 것이므로 전체 데이터에 대한 파악이 중요할 뿐만 아니라 해당 비즈니스 도메인에 대한 충분한 이해가 수반되어야 함
    - 예: 전주 대비 방문 횟수 증감률, 전년도 대비 클릭 횟수 증감률 등의 파생 변수를 만들어서 예측력을 높임
    - 파생변수는 무작정 변수를 가공해서 만드는 것이 아니라 데이터의 특성과 흐름을 충분히 파악 → 아이디어를 얻어서 만들어야 함!

예시:

![image](https://github.com/user-attachments/assets/92bbdf92-159a-47c8-8957-33abd0ab2b3e)
![image](https://github.com/user-attachments/assets/30d61903-1f4d-44a2-8bd4-001fc1492c3f)


### 11.5.1 파생 변수 생성 실습

```python
# 두 개의 변수 결합한 파생변수 생성

# 구매 상품당 가격 컬럼 생성
df['Unit_amount'] = df['Sales_Amount']/df['Quantity']

# 총 구매가격 컬럼 생성
df['All_amount'] = \
df[['Quantity', 'Sales_Amount']].apply(lambda series: series.prod(), axis=1)

df.tail()
```

```python
# 로그, 제곱근, 제곱 변환 파생변수 생성

# 방법1.Sales_Amount 컬럼 로그 적용 (+1)
df['Sales_Amount_log'] = preprocessing.scale(np.log(df['Sales_Amount']+1))

# 방법2.Sales_Amount 컬럼 로그 적용 (+1)
df['Sales_Amount_log2'] = df[['Sales_Amount']].apply(lambda x: np.log(x+1))    

# Sales_Amount 컬럼 제곱근 적용 (+1)
df['Sales_Amount_sqrt'] = np.sqrt(df['Sales_Amount']+1)

# Sales_Amount 컬럼 제곱 적용
df['Sales_Amount_pow'] = pow(df[['Sales_Amount']],2)

df.tail()
```

```python
# 월 합계, 평균 구매금액 변수 생성

# date 컬럼 날짜형식 변환
df['Date2']= pd.to_datetime(df['Date'], infer_datetime_format=True) 

# 연도 컬럼 생성
df['Year'] = df['Date2'].dt.year

# 월 컬럼 생성
df['Month'] = df['Date2'].dt.month

#연월별, 고객별 매출 합계, 평균 컬럼 생성
df_sm = df.groupby(['Year', 
                    'Month', 
                    'Customer_ID'])['Sales_Amount'].agg(['sum','mean']).reset_index()

# 기존 일별 테이블에 평균 테이블 조인
df2 = pd.merge(df, df_sm, how='left')

df2.head()
```

```python
# 월 평균 구매금액 대비 일 별 구매금액 차이 변수 생성
df2['Sales_Amount_Diff'] = df2['mean'] - df2['Sales_Amount']

# 월 평균 구매금액 대비 일 별 구매금액 비율 변수 생성
df2['Sales_Amount_UD'] = df2['Sales_Amount'] / df2['mean']
    
# 월 총 구매금액 대비 일 별 구매금액 비율 변수 생성
df2['Sales_Amount_Rto'] = df2['Sales_Amount']/df2['sum']
    
df2.head()
```

```python
# 전 월 값 파생변수 생성

# 4주 뒤 시점 컬럼 생성
df2['Date2_1_m'] = df2['Date2'] + timedelta(weeks=4)

# # 4주 뒤 시점연도 컬럼 생성
df['Year_1_m'] = df2['Date2_1_m'].dt.year

# # 4주 뒤 시점월 컬럼 생성
df['Month_1_m'] = df2['Date2_1_m'].dt.month

# 4주 전 구매금액 연월별, 고객별 매출 평균 컬럼 생성
df_Mn_1 = df.groupby(['Year_1_m', 
                      'Month_1_m', 
                      'Customer_ID'])['Sales_Amount'].agg(['sum',
                                                           'mean']).reset_index()

# 조인을 위한 컬럼명 변경 
df_Mn_1.rename(columns={'Year_1_m':'Year', 
                        'Month_1_m':'Month', 
                        'sum':"sum_1_m", 
                        'mean':'mean_1_m'}, inplace=True)

df2 = pd.merge(df2, df_Mn_1, how='left')

df2.head()
```

```python
# 전 월과의 차이 파생변수 생성

# 전 월 대비 구매금액 평균 차이 변수 생성
df2['Mn_diff_1_mean'] = df2['mean'] - df2['mean_1_m']

# 전 월 대비 총 구매금액 차이 변수 생성
df2['Mn_diff_1_sum'] = df2['sum'] - df2['sum_1_m']

df2.head()
```

# 확인 문제

## 문제 1. 데이터 전처리

> 🧚 한 금융회사의 대출 데이터에서 소득 변수에 결측치가 포함되어 있다. 다음 중 가장 적절한 결측치 처리 방법은 무엇인가?
> 

> [보기]
1️⃣ 결측값이 포함된 행을 모두 제거한다.
2️⃣ 결측값을 소득 변수의 평균값으로 대체한다.
3️⃣ 연령과 직업군을 독립변수로 사용하여 회귀 모델을 만들어 소득 값을 예측한다.
4️⃣ 결측값을 보간법을 이용해 채운다.
> 

> [데이터 특징]
> 
> 
> - `소득` 변수는 연속형 변수이다.
> 
> - 소득과 `연령`, `직업군` 간에 강한 상관관계가 있다.
> 
> - 데이터셋에서 `소득` 변수의 결측 비율은 15%이다.
> 

```
소득 변수가 연속형이고, 10%가 넘기 때문에 결측값이 포함된 행을 모두 제거하는 것을 옳지 않다.또한 시계열 데이터도 아니므로 보간법도 알맞지 않다. 소득 변수의 평균값으로 대체하면 분산이 너무 작아지므로, 소득과 강한 상관관계가 있는 직업군으로 회귀 모델을 만들어 소득값을 예측하는 것이 적절해보인다.

```

## 문제 2. 데이터 스케일링

> 🧚 머신러닝 모델을 학습하는 과정에서, 연봉(단위: 원)과 근속연수(단위: 년)를 동시에 독립변수로 사용해야 합니다. 연봉과 근속연수를 같은 스케일로 맞추기 위해 어떤 스케일링 기법을 적용하는 것이 더 적절한가요?
> 

<!--표준화와 정규화의 차이점에 대해 고민해보세요.-->

```
연봉과 근속년수 단위차이가 많이 나기 때문에 평균이 0이고 표준편차가 1인 표준화가 더 적절해보인다.
```

### 🎉 수고하셨습니다.
