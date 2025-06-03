# 통계학 7주차 정규과제

📌통계학 정규과제는 매주 정해진 분량의 『*데이터 분석가가 반드시 알아야 할 모든 것*』 을 읽고 학습하는 것입니다. 이번 주는 아래의 **Statistics_7th_TIL**에 나열된 분량을 읽고 `학습 목표`에 맞게 공부하시면 됩니다.

아래의 문제를 풀어보며 학습 내용을 점검하세요. 문제를 해결하는 과정에서 개념을 스스로 정리하고, 필요한 경우 추가자료와 교재를 다시 참고하여 보완하는 것이 좋습니다.

7주차는 `3부. 데이터 분석하기`를 읽고 새롭게 배운 내용을 정리해주시면 됩니다.

## Statistics_7th_TIL

### 3부. 데이터 분석하기

### 13.머신러닝 분석 방법론

### 14.모델 평가

## Study Schedule

| 주차 | 공부 범위 | 완료 여부 |
| --- | --- | --- |
| 1주차 | 1부 p.2~56 | ✅ |
| 2주차 | 1부 p.57~79 | ✅ |
| 3주차 | 2부 p.82~120 | ✅ |
| 4주차 | 2부 p.121~202 | ✅ |
| 5주차 | 2부 p.203~254 | ✅ |
| 6주차 | 3부 p.300~356 | ✅ |
| 7주차 | 3부 p.357~615 | ✅ |

<!-- 여기까진 그대로 둬 주세요-->

# 13.머신러닝 분석 방법론

```
✅ 학습 목표 :
* 선형 회귀와 다항 회귀를 비교하고, 데이터를 활용하여 적절한 회귀 모델을 구축할 수 있다.
* 로지스틱 회귀 분석의 개념과 오즈(Odds)의 의미를 설명하고, 분류 문제에 적용할 수 있다.
* k-means 알고리즘의 원리를 설명하고, 적절한 군집 개수를 결정하여 데이터를 군집화할 수 있다.

```

## 13.1. 선형 회귀분석과 Elastic Net(예측모델)

### 13.1.1 회귀분석의 기원과 원리

**💌 회귀분석**

- 종속변수 Y의 값에 영향을 주는 독립 변수 X들의 조건을 고려하여 구한 평균값
- 절편, 기울기, 오차항
- 오차항: 독립변수가 종속변수에 주는 영향력을 제외한 변화량
- 오차항을 최소화하는 절편과 기울기를 구하는 것

![image](https://github.com/user-attachments/assets/c986737c-4187-4d53-b004-3dfef8efa4a1)

![image](https://github.com/user-attachments/assets/b83a03c9-ced1-4df8-9e49-8656cd018c94)

**💥 최적의 회귀선은 어떻게 구할까?**

- 모형적합
- 회귀선과 각 관측치를 뜻하는 점 간의 거리를 최소화
- 예측치와 관측치들 간의 수직 거리(오차)의 제곱합을 최소로 하는 직선 = 회귀선

⇒ 최소제곱추정법

- 최대우도 추정법
    - 변수x1,x2 … xn이 주어졌을 때 예측값 y hat이 관측될 가능도를 최대화
- 단순 회귀 분석, 단변량 회귀분석
- 다중 회귀 분석, 다변량 회귀분석
    - 다중공선성
    - 잔차의 정규성
        - Q-Q plot이 대략적으로 직선 형태를 보일 때 따른다고 판단
        - 샤피로-위클 검정이나 앤더슨-달링 검정 사용
    - 잔차의 등분산성
    - 독립성
    - 선형성
- 비선형의 관계일 경우 예측력이 떨어지는 문제 발생
    - 방지하기 위해 변수를 구간화하여 더미 변수로 변환하여 분석
    - 로그함수로 치환하면 비선형성을 완화시킬 수 있음

### 13.1.2 다항회귀

- 독립변수와 종속변수가 곡선일 경우 일반 회귀모델은 한계가 있기에 고안된 방법
- 독립변수와 종속변수가 곡선형 관계일 때, 변수에 **각 특성의 제곱을 추가하여** 회귀선을 곡선형으로 변환
- 단, 차수가 커질수록 편향은 감소하지만 변동성이 커짐 ⇒ 분산↑, 과적합 유발

![image](https://github.com/user-attachments/assets/a7778c2f-16d5-4672-8679-cc7266d766e2)

- 회귀 분석의 기본 가설
    - 귀무가설: 모든 회귀 계수 =0
    - 대립가설: 적어도 하나의 회귀 계수 ≠0

예시

![image](https://github.com/user-attachments/assets/4d1375e5-4f2d-47f3-a1e8-30ec2dbdcc2e)

- x1계수=0.604
    - 독립변수x1가 1씩 커질 때마다 종속 변수 y값이 0.604만큼 커진다는 것을 의미
- intercept=24.822
    - 절편, 종속 변수의 기본값
- standard error=표준오차
    - 값이 크다는 것= 예측값과 실젯값의 차이가 크다
    - 변수마다 스케일 차이가 다르므로 표준오차의 절댓값만으로 실제 오차가 큰지 판단하기 어려움
    
    → t-value 사용
    
    - 노이즈 대비 시그널의 강도
    - 독립변수와 종속변수 간에 선형관계가 얼마나 강한지 나타내기 때문에 값이 커야함
    - but 일일이 조합해서 판단하는 건 비효율적..
    
    **→ p-value 사용**
    
- tolerance, VIF
    - 해당 변수의 다른 독립 변수들과의 상관관계 수준을 판단하는 기준

```
**📢 but, 분석가가 수동으로 변수 조합을 테스트하는 것은 비효율적이다 
⇒ 조합을 자동으로 선택할 수 있는 변수 선택 알고리즘**
```

1. **전진 선택법**
    - 절편만 있는 모델에서 시작
    - 유의미한 독립변수 순으로 변수 하나씩 추가
    - 이전 변수 집합에 비해 새로운 변수를 추가했을 때 모델 적합도가 기준치 이상 증가하지 못할 경우, 변수 선택 종료
    - 빠르지만 한번 선택된 변수는 다시 제거되지 않음
2. **후진 제거법**
    - 모든 독립 변수가 있는 상태에서 유의미하지 않은 순으로 설명 변수 하나씩 제거
    - 어느 한 변수를 제거했을 때, 모델 적합도가 기준치 이상 감소하는 경우 더 이상 변수를 제거하지 않음
    - 한번 제거된 변수는 추가되지 않음
    - 시간이 오래 걸리지만 안전함
3. **단계적 선택법**
    - 변수를 하나씩 추가하면서 선택된 변수가 3개 이상이 되면 변수 추가와 제거를 번갈아가며 수행
    - 선택된 독립 변수 모델의 잔차를 구하여 선택되지 않은 나머지 변수와의 잔차 상관도 ⇒ 변수 선택
    - 오래 걸림

## 13.2. 로지스틱 회귀분석 (분류모델)

- 종속변수가 질적 척도
- 특정 수치를 예측하는 것이 아니라 어떤 카테고리에 들어갈지 분류하는 모델
- 종속변수가 이항으로 이뤄져 예측
- 종속변수의 범주가 3개 이상일 경우에는 다항 로지스틱 회귀분석을 통해 분류 예측

![image](https://github.com/user-attachments/assets/8d533375-1944-46aa-9e8f-c6b95845b139)

- 종속변수의 값= 1이 될 확률
- 0.5보다 크면 1, 작으면 0으로 분류
- 오즈 값 구하기
    - 사건이 발생한 가능성이 발생하지 않을 가능성보다 어느 정도 큰지 나타내는 값
    
    ![image](https://github.com/user-attachments/assets/f1f5598e-2cad-4dbc-8489-d412106fd7c8)

    
- 발생확률이 1에 가까워질수록 오즈 값은 기하급수적으로 커지고 최솟값은 0이 됨

![image](https://github.com/user-attachments/assets/65474dc8-283a-4651-b68d-4766b4cf6e77)

- 여전히 0에서 1사이의 범위를 나타내지 못하는 문제 → 로짓 변환하여 0에서 1사이로 치환

⇒ 시그모이드 함수

![image](https://github.com/user-attachments/assets/55fa59fa-34c9-41f0-bc99-c05b4020cb29)

- 범주가 여러 개더라도 각 범주마다 이항 로지스틱을 시행하여 확률 구한다.
    - 하나의 범주를 기준으로 잡고 나머지 다른 범주들과 비교해서 식 만들기
- 선형 회귀분석과 달리 각 변수의 오즈비를 알 수 있음
- R^2를 구하는 방식이 여러개

### 13.2.1 로지스틱 회귀분석 실습

```
# 명목형 변수 더미처리

# 하나의 가변수 범주 제거 옵션 적용
df2 = pd.get_dummies(df, columns = ['HeartDisease','Smoking',
                                    'AlcoholDrinking','Stroke',
                                    'DiffWalking','Sex',
                                    'AgeCategory','Race',
                                    'Diabetic','PhysicalActivity',
                                    'GenHealth','Asthma',
                                    'KidneyDisease','SkinCancer']
                     ,drop_first=True
                    )

df2.head()
```

![image](https://github.com/user-attachments/assets/bf6d125f-ee7f-4100-bcc5-3baea85f6f7b)

```
# 독립변수와 종속변수 분리하여 생성
X = df3.drop(['HeartDisease_Yes'],axis=1)
y = df3[['HeartDisease_Yes']]

# 학습셋과 테스트셋 분리하여 생성(7.5:2.5)
X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.25,random_state=10)

# 학습셋과 검증셋이 잘 나뉘었는지 확인
print('train data 개수: ', len(X_train))
print('test data 개수: ', len(X_test))
```

![image](https://github.com/user-attachments/assets/d45da8e1-271a-4434-9fe9-f4294ad5e213)

```
sns.countplot(x="HeartDisease_Yes", data=y_train)

plt.show()
```

![image](https://github.com/user-attachments/assets/312220d1-821f-49f8-a38a-e8374773cc11)

```
X_train_re = X_train.copy()
y_train_re = y_train.copy()

X_temp_name = ['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10',
            'X11','X12','X13','X14','X15','X16','X17','X18','X19','X20',
            'X21','X22','X23','X24','X25','X26','X27','X28','X29','X30',
            'X31','X32','X33','X34','X35','X36','X37']
y_temp_name = ['y1']

X_train_re.columns = X_temp_name
y_train_re.columns = y_temp_name

X_train_re.head()
```
![image](https://github.com/user-attachments/assets/1cad5cec-710d-4cc3-967e-168217f6453e)


```
X_train_under, y_train_under = RandomUnderSampler(
    random_state=0).fit_resample(X_train_re,y_train_re)

print('RandomUnderSampler 적용 전 학습셋 변수/레이블 데이터 세트: '
      , X_train_re.shape, y_train_re.shape)
print('RandomUnderSampler 적용 후 학습셋 변수/레이블 데이터 세트: '
      , X_train_under.shape, y_train_under.shape)
print('RandomUnderSampler 적용 전 레이블 값 분포: \n'
      , pd.Series(y_train_re['y1']).value_counts())
print('RandomUnderSampler 적용 후 레이블 값 분포: \n'
      , pd.Series(y_train_under['y1']).value_counts())
```
![image](https://github.com/user-attachments/assets/756c2160-0c0f-46e5-a032-11a5d2880ba8)

```
X_train_under.columns = list(X_train)
y_train_under.columns = list(y_train)

X_train_under.head()
```
![image](https://github.com/user-attachments/assets/37c8e40b-f64c-4400-82e0-1d6698453dbf)

```
model = LogisticRegression()
model.fit(X_train_under, y_train_under)

print('학습셋 모델 정확도:', model.score(X_train_under, y_train_under))
```
![image](https://github.com/user-attachments/assets/fbb46eab-132f-4382-b5c8-be008eeb3bf6)


```
print('테스트셋 모델 정확도:', model.score(X_test, y_test))
```
![image](https://github.com/user-attachments/assets/53cd923c-2426-469a-b018-cd7529e8d8f2)

```
print(model.coef_)
```
![image](https://github.com/user-attachments/assets/9c571754-ff25-4f89-95f7-28746aedf27c)


```
model2 = sm.Logit(y_train_under, X_train_under)
results = model2.fit(method = "newton") 

results.summary()
```
![image](https://github.com/user-attachments/assets/a7ed62cd-8d19-447e-9aa4-0e66b98d7c6d)
![image](https://github.com/user-attachments/assets/396f9c51-4df3-47c3-b2e7-4f0706071b21)


```
np.exp(results.params)
```

## 13.8. k-means 클러스터링(군집모델)

- 비지도 학습
- 미리 가지고 있는 레이블 없이 데이터의 특성과 구조를 발견
- 구현 방법이 매우 간단하고 실행 속도가 빠름
- k: 분류할 군집의 수
- 중심점과 군집 내 관측치 간의 거리를 비용 함수로 하여, 이 함수 값이 최소화되도록 중심점과 군집을 반복적으로 재정의
- → k개의 중심점을 찍어서 관측치들 간의 거리를 최적화하여 군집화하는 모델 `k-means 클러스터링`
- 단계
    1. k개의 중심점을 임의의 데이터 공간에 선정
    2. 각 중심점과 관측치들 간의 유클리드 거리를 계산
    3. 각 중심점과 거리가 가까운 관측치들을 해당 군집으로 할당
    4. 할당된 군집의 관측치들과 해당 중심점과의 유클리드 거리를 계산
    5. 중심점을 군집의 중앙으로 이동(군집의 관측치들 간 거리 최소 지점)
    6. 중심점이 더 이상 이동하지 않을 때까지 2~5단계 반복
    
    ![image](https://github.com/user-attachments/assets/216f8f95-7e6f-4753-ba7a-311a48388586)

    
- local minimum 현상
    - (3)번과 같은 위쪽과 아래쪽으로 군집화가 되는 일 발생

![image](https://github.com/user-attachments/assets/db4e1a8e-a4d5-41a2-a1ee-557b0cee233f)


- 함수 값이 최소화되도록 중심점과 군집을 반복적으로 재정의
- 관측치의 거리 합이 최소화됐을 때 클러스터링 알고리즘이 종료되기 때문에 거리합이 최소화되는 전역 최솟값을 찾기 전에 지역 최솟값에서 알고리즘 종료됨
- **지역 최솟값 문제를 방지하기 위해 초기 중심점 선정 방법을 다양하게 하여 최적의 모델 선정!**

![image](https://github.com/user-attachments/assets/4a7dab54-e7ed-4dcf-9eb7-02bd9b78fd55)

- 적절한 k의 수는 어떻게 선정할까
    - 비즈니스 도메인 지식을 통한 개수 선정
    - 엘보우 기법: 군집 내 중심점과 관측치 간 거리 합이 급감하는 구간의 k개수 선정
    - 실루엣 계수: 군집 안의 관측치들이 다른 군집과 비교해서 얼마나 비슷한지 나타내는 수치

![image](https://github.com/user-attachments/assets/483f12b5-a838-4728-9f1c-a33c0961d496)

- k값이 적절해도 데이터 형태가 적합하지 않으면 효과적인 군집화를 할 수 없다.

![image](https://github.com/user-attachments/assets/6a772a4a-8160-4299-bb0d-2611cf761c3a)


- k-means는 중심점과 관측치 간의 거리를 이용하므로 이러한 데이터는 효과적으로 분류할 수 없다

⇒ 밀도 기반의 원리를 이용한 DBSCAN 클러스터링 기법

**💫 DBSCAN**

- 별도의 k수 지정이 필요 없음
- 기준 관측치로부터 이웃한 관측치인지 구별할 수 있는 거리 기준이 필요
    - 거리 기준값이 크면 데이터 공간상 멀리 있는 관측치도 이웃한 관측치로 인식
- 거리 기준 내에 포함된 이웃 관측치 수에 대한 기준 필요
    - 특정 거리 안에 몇 개의 이상의 관측치가 있어야 하나의 군집으로 판단할 것인가 결정
- U자형, H자형 데이터 분포도 효과적으로 군집화할 수 있음
- k-means 방식에 비해 분류에 필요한 연상량이 많음 → 변수를 적절히 설정해줘야 한다.
- 데이터 특성을 모를 경우에는 적절한 파라미터 값을 설정하는 것이 어렵다

![image](https://github.com/user-attachments/assets/9d41e3a1-d1ec-48af-b7e8-e4aeb38b3401)

- k-means 클러스터링에 사용한 독립변수들이 다른 군집들에 비해서 어떠한 특성을 가지고 있는지를 확인하여 각 군집을 명확하게 정의해야한다.

### 13.8.1 k-means 클러스터링 실습

```
# 산점도 행렬 시각화

# ID 컬럼 제거
df1 = df.drop('CustomerID', axis=1)

sns.set(font_scale=1) 
sns.set_style('ticks') 
sns.pairplot(df1, 
             diag_kind='kde', # 상관계수가 1이면 분포로 표시
             hue = 'Gender',
             corner=True,
             height = 5
            )
plot_kws={"s": 100}

plt.show()
```

![image](https://github.com/user-attachments/assets/35d53f44-3325-461b-8acc-748e2e136adf)

```
# 데이터 스케일 정규화

# Gender변수 가변수 처리
df2 = pd.get_dummies(df1, columns = ['Gender'],drop_first=True)

# 데이터 정규화 적용
MinMaxScaler = MinMaxScaler()
df_minmax = MinMaxScaler.fit_transform(df2)

# 컬럼명 결합
df_minmax = pd.DataFrame(data=df_minmax, columns=df2.columns)

df_minmax.head()
```

![image](https://github.com/user-attachments/assets/5215fc75-6e2a-402e-ae6f-a0ef0b1869ef)

```
# k-means 클러스터링 모델 생성

kmeans_model_1 = KMeans(
    init="k-means++",
    n_clusters=3,
    n_init=10,
    max_iter=300,
    random_state=37
)

kmeans_model_1.fit(df_minmax)

# 최저 SSE 값
print(kmeans_model_1.inertia_)

# 각 군집의 중심점 좌표 확인
print(kmeans_model_1.cluster_centers_)

# 반복 횟수
print(kmeans_model_1.n_iter_)
```

![image](https://github.com/user-attachments/assets/7e729e8d-f510-41f8-955b-c9b2a3a3424a)


```
# 엘보우 차트 시각화

Elbow_Chart = KElbowVisualizer(kmeans_model_1, k=(1,11),)
Elbow_Chart.fit(df_minmax)
Elbow_Chart.draw() 
```

![image](https://github.com/user-attachments/assets/2e865ec6-9b1d-4d53-a647-6ae1051f4961)


```
# 실루엣 계수 시각화 1

# k-means 모델 설정
kmeans_model_2 = {
       "init": "k-means++",
       "n_init": 10,
       "max_iter": 300,
       "random_state": 37,
        }

# 각 K의 실루엣 계수 저장
silhouette_coef = []

# 실루엣 계수 그래프 생성
for k in range(2, 11):
    kmeans_silhouette = KMeans(n_clusters=k, **kmeans_model_2)
    kmeans_silhouette.fit(df_minmax)
    score = silhouette_score(df_minmax, kmeans_silhouette.labels_)
    silhouette_coef.append(score)
    
plt.style.use('seaborn-whitegrid')
plt.plot(range(2, 11), silhouette_coef)
plt.xticks(range(2, 11))
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Coefficient')
plt.show()
```

![image](https://github.com/user-attachments/assets/e78d86b6-ef4e-4a4a-8dde-e56bfe0a1798)


```
# 실루엣 계수 시각화 2

fig, ax = plt.subplots(3, 2, figsize=(15,15))
for i in [2, 3, 4, 5, 6, 7]:

# k-means 클러스터링 모델 생성
    kmeans_model_3 = KMeans(n_clusters=i, 
                            init="k-means++", 
                            n_init=10, 
                            max_iter=300, 
                            random_state=37)
    q, mod = divmod(i, 2)

# 실루엣 계수 시각화    
    visualizer = SilhouetteVisualizer(kmeans_model_3, 
                                      colors="yellowbrick", 
                                      ax=ax[q-1][mod])
    visualizer.fit(df_minmax)
```

![image](https://github.com/user-attachments/assets/66a8957d-0464-417b-a405-a1bebee7c565)


```
# k-means 클러스터 시각화

# k-means 모델 설정
kmeans_model_4 = KMeans(
    init="k-means++",
    n_clusters=4,
    n_init=10,
    max_iter=300,
    random_state=37)

# 군집 번호 결합 
df2['cluster1'] = kmeans_model_4.fit_predict(df_minmax)

# 시각화 설정
plt.figure(figsize = (8, 8))
for i in range(0, df2['cluster1'].max() + 1):
    plt.scatter(df2.loc[df2['cluster1'] == i, 
                        'Annual Income (k$)'], 
                df2.loc[df2['cluster1'] == i, 
                        'Spending Score (1-100)'], 
                label = 'cluster'+str(i))

plt.legend()
plt.title('K means visualization', size = 12)
plt.xlabel('Annual Income (k$)', size = 10)
plt.ylabel('Spending Score (1-100)', size = 10)
plt.show()
```

![image](https://github.com/user-attachments/assets/42f4aab4-1f95-4a71-a303-c969d7e02a32)


```
# DBSCAN 모델 생성 및 시각화

# DBSCAN 모델 설정
DBSCAN_model = DBSCAN(eps=0.7, min_samples=5)

# 군집화 모델 학습 및 클러스터 예측 결과 반환
DBSCAN_model.fit(df_minmax)
df2['cluster2'] = DBSCAN_model.fit_predict(df_minmax)

# 시각화 설정
plt.figure(figsize = (8, 8))
for i in range(0, df2['cluster2'].max() + 1):
    plt.scatter(df2.loc[df2['cluster2'] == i, 
                        'Annual Income (k$)'], 
                df2.loc[df2['cluster2'] == i, 
                        'Spending Score (1-100)'], 
                    label = 'cluster'+str(i))

plt.legend()
plt.title('DBSCAN visualization', size = 12)
plt.xlabel('Annual Income (k$)', size = 10)
plt.ylabel('Spending Score (1-100)', size = 10)
plt.show()
```

![image](https://github.com/user-attachments/assets/6f4b0421-1e41-43a1-897b-2b898dfd8e16)

```
# k-means 군집 별 특성 확인

df_kmeans = df2.groupby(['cluster1']).agg({'Age':'mean',
                                          'Annual Income (k$)':'mean',
                                          'Spending Score (1-100)':'mean',
                                          'Gender_Male':'mean'
                                          }).reset_index()

df_kmeans['cnt'] = df2.groupby('cluster1')['Age'].count()
df_kmeans.head()
```

![image](https://github.com/user-attachments/assets/c4212a60-7d73-44b5-bec6-88cfcdd3aa29)


```
# DBSCAN 군집 별 특성 확인

df_DBSCAN = df2.groupby(['cluster2']).agg({'Age':'mean',
                                          'Annual Income (k$)':'mean',
                                          'Spending Score (1-100)':'mean',
                                          'Gender_Male':'mean'}).reset_index()

df_DBSCAN['cnt'] = df2.groupby('cluster2')['Age'].count()
df_DBSCAN.head()
```

![image](https://github.com/user-attachments/assets/9a900436-065e-4837-9594-ff942793c0d9)


# 14. 모델 평가

```
✅ 학습 목표 :
* 유의확률(p-value)을 해석할 때 주의할 점을 설명할 수 있다.
* 분석가가 올바른 주관적 판단을 위한 필수 요소를 식별할 수 있다.

```

## 14.3. 회귀성능 평가지표

### 14.3.1 R-Square와 Adjusted R-Sqare

- 회귀 모델의 회귀선이 종속변수 y값을 얼마나 잘 설명할 수 있는가를 의미

![image](https://github.com/user-attachments/assets/e7adb4cd-ac5e-4dae-9d01-07ccad51d31c)

- SST: 회귀식의 추정값과 전체 실젯값 평균과의 편차 제곱합
- SSE: 실젯값과 전체 실젯값 평균과의 편차 제곱합

⇒ R^2는 SSR값이 클수록 SST 값이 작을수록 커짐

SST값이 크다 = 회귀선이 각 데이터를 고르게 설명한다

SST값이 작다 = SSR을 제외한 SSE값이 작다 → 모델의 설명력이 높아진다.

SSR↑ SST↓ R-Square ↑

- Adjusted R-Square
    - R^2가 독립 변수의 개수가 많아질수록 값이 커지는 문제를 보정한 기준

![image](https://github.com/user-attachments/assets/19ebf441-c5b1-485e-9881-c0f9fd430529)

### 14.3.2 RMSE(root mean square error)

- 수치를 정확히 맞춘 비율이 아닌, 실제 수치와의 차이를 회귀 모델 지표로 사용

![image](https://github.com/user-attachments/assets/120fe6d9-357f-4d7c-aff0-a5e5d53d1b18)

- 실제 수치와 예측한 수치와의 차이를 확인하는 방법
- 실젯값과 예측값의 표준편차를 구하는 것

### 14.3.3 MAE(mean absolute error)

- MAE는 실젯값과 예측값의 차이 절댓값 합을 n으로 나눈 값

![image](https://github.com/user-attachments/assets/0ac8780d-33e3-4e3d-af64-0309a591b5b4)

- RMSE: 평균 제곱 오차
- MAE: 평균 절대 오차

### 14.3.4 MAPE(mean absolute percentage error)

- 백분율 오차인 MAPE: MAE를 퍼센트로 변환
- MAPE
    - 0부터 무한대의 값을 가질 수 있음
    - 0에 가까울수록 우수한 모델
    
    ![image](https://github.com/user-attachments/assets/f4189d12-4d02-4e80-8263-b7137259d2bc)

    
- 주의해야 할 점
    - 실젯값이 0인 경우에는 0으로 나눌 수 없기 때문에 MAPE를 구할 수 없다
        
        → 실젯값에 0이 많은 데이터는 MAPE 평가 기준을 사용하는 것은 적합하지 않다
        
    - 실젯값이 양수인 경우
        - 실젯값보다 작은 값으로 예측하는 경우: MAPE의 최댓값이 최대 100%까지만 커질 수 있다.
    - 실젯값보다 크게 예측하는 경우
        - MAPE값이 한계가 없기 때문에 MAPE 기준으로 모델을 학습하면 실젯값보다 작은 값으로 예측하도록 편향될 수 있음
    - 실젯값이 0과 가까운 매우 작은 값인 경우
        - MAPE가 과도하게 높아지는 경우가 발생할 수 있다

### 14.3.5 RMSLE(root mean square logarithmic error)

- RMSLE: RMSE와 동일한 수식에서 실젯값과 예측값에 1을 더해준 다음 로그를 취해준 평가 방식
- 로그를 취하면서 상대적 비율 비교 가능 ⇒ RMSLE가 오차 이상치에 덜 민감함
    - 실젯값에 이상치가 존재하는 경우에 적절
    - 로그를 취하기 전에 1을 더하는 이유는 실젯값이 0일 때 log0이 무한대로 수렴할 수 있음

![image](https://github.com/user-attachments/assets/7544edcf-7fe2-4ecf-9617-6775d7c953d4)

- MAPE: 스케일이 차이가 나더라도 오차 비율이 같으면 동일한 RMSLE값 산출

### 14.3.6 AIC와 BIC

- AIC
    - 최대 우도에 독립변수가 얼마나 많은가에 따른 페널티를 반영하여 계산하는 모델 평가 척도
    - 모델의 정확도와 함께 독립변수가 많고 적음까지 따져서 우수한 모델을 선택할 수 있도록 하는 평가 방법
    - AIC가 작을수록 좋은 모델, 변수가 적을수록 AIC 작아짐
    - 변수가 늘어날수록 모델의 편의 감소, 분산 증가

![image](https://github.com/user-attachments/assets/0769e2a0-1840-47f6-bc74-db21bab0a9cf)

![image](https://github.com/user-attachments/assets/e8b723d1-274a-430a-910a-7f7dd792affb)

- 관측치가 적은 겨웅에는 관측치 수를 반영하여 보정된 AICc 방식을 사용할 수 있다

![image](https://github.com/user-attachments/assets/522bdbeb-5dac-4b26-a1ff-5d3b0a84af69)

- BIC
    - 변수의 개수를 줄이는 것을 중요하게 여기는 상황에서는 모델을 평가

![image](https://github.com/user-attachments/assets/d2d54497-7d88-4ee5-ab83-afb8d34fe362)


### 14.3.7 회귀성능 평가지표 실습

```
# 독립변수와 종속변수 분리하여 생성
x = df[[ 'bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 
       'sqft_living15', 'sqft_lot15']]
# 'id', 'date'는 키값에 해당하므로 변수에서 제외 해준다.
y = df[['price']]

# 학습셋과 테스트셋 분리하여 생성(7:3)
# df_train, df_test = train_test_split(df, test_size = 0.4) 
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.6, test_size=0.4)
```

```
# 다중회귀모델 생성
mreg = LinearRegression(fit_intercept=True)
mreg.fit(x_train, y_train)

# 테스트셋에 모델 적용
y_predict = mreg.predict(x_test)
```

```
# RMSE 산출 (MSE에 루트 적용)
MSE = mean_squared_error(y_test, y_predict)
RMSE = np.sqrt(MSE)
print(("RMSE : {:.2f}".format(RMSE)))

# MAE 산출
MAE = mean_absolute_error(y_test, y_predict)
print(("MAE : {:.2f}".format(MAE)))

# MAPE 산출
MAPE = mean_absolute_percentage_error(y_test, y_predict)
print(("MAPE : {:.2f}".format(MAPE)))

# RMSLE 산출 (MSLE에 루트 적용)

# 음수값 전처리
y_predict_df = pd.DataFrame(y_predict,columns=['price2'])
y_predict_df2 = y_predict_df.copy()
y_predict_df2.loc[y_predict_df2['price2'] < 0, 'price2'] = 0
y_predict_rmsle = y_predict_df2.to_numpy()

MSLE = mean_squared_log_error(y_test, (y_predict_rmsle))
RMSLE = np.sqrt(MSLE)
print(("RMSLE : {:.2f}".format(RMSLE)))
```

![image](https://github.com/user-attachments/assets/b97fe4af-5c6a-427e-a197-90fcf4962264)


```
# RMSLE ver. 2
  
def rmsle(predicted_values, actual_values):
    
    # 테스트셋 y 값과 예측값에 +1 및 로그 
    log_y_test = np.log(y_test + 1)
    log_y_predict = np.log(y_predict + 1)

    # 테스트셋 y 값 - 예측값 및 제곱
    diff = log_y_predict - log_y_test
    diff_square = np.square(diff)

    # 차이값 평균 및 루트
    mean_diff = diff_square.mean()
    final_rmsle = np.sqrt(mean_diff)  

    return final_rmsle

rmsle(y_test, y_predict)
```

![image](https://github.com/user-attachments/assets/c70781f9-5576-4114-a4da-0abaf46df758)


## 14.6. 유의확률의 함정

- 유의확률은 정말 절대적으로 확실한 기준값
- 대부분 p값이 0.05 이하로 나오면 유의하다고 판단
- p값: 표본의 크기 🔼 p값 🔽 (0에 수렴)
    - 표본의 크기가 커지면 표본 오차가 작아짐 → p값도 작아짐
- 0.05도 통상적으로 쓰이는 임의적인 기준이므로 0.05 미만으로 나왔다고 해도 통계적 유의성이 확실하다고 말할 수 x

## 14.7. 분석가의 주관적 판단과 스토리텔링

- 아무리 AI, 딥러닝이 발전했다 해도, 데이터 분석가의 역할은 중요하다
- 사람의 인사이트와 결정적 판단 매우 중요 !!
- EX.1
    - c 차종의 결과 의문 : 50-60대 남성의 구매 예측 확률 높게 나옴
    - 첫 차를 구매하는 시기에 부모나 할아버지가 자식이나 손자, 손녀의 자동차를 대신 구입해주는 문화 존재
- EX.2
    - 이탈 고객 방어 프로모션
    - 프로모션한 고객의 이탈률이 더 높았음 → 약정 기간을 꼼꼼하게 확인하고 있지 않다가 약정 기간 끝났다는 프로모션 연락을 받고 상기(트리거 역할)
- 올바른 주관적 판단을 하기 위해서…
    - 해당 분야의 도메인 지식 수반
    - 통계적 지식을 기반으로 EDA와 데이터 전처리 성실히 수행
- 적극적인 커뮤니케이션과 검증 과정이 필요
    - 결과가 나왔을 때 바로 현실 적용하기 보단 사전 검증하는 것이 필요
- 타인을 이해시키고 설득시킬 수 있는 스토리텔링 구조
    - 배경 - 문제(위기) - 극복 - 변화

![image](https://github.com/user-attachments/assets/9a281532-2c59-41e0-9ff2-a3f0aef97452)


# 확인 문제

## **문제 1. 선형 회귀**

> 🧚 칼 피어슨의 아버지와 아들의 키 연구 결과를 바탕으로, 다음 선형 회귀식을 해석하세요.
> 
> 
> 칼 피어슨(Karl Pearson)은 아버지(X)와 아들(Y)의 키를 조사한 결과를 바탕으로 아래와 같은 선형 회귀식을 도출하였습니다. 아래의 선형 회귀식을 보고 기울기의 의미를 설명하세요.
> 
> **ŷ = 33.73 + 0.516X**
> 
> - **X**: 아버지의 키 (cm)
> - **ŷ**: 아들의 예상 키 (cm)

```
아버지의 키가 1cm클수록 아들의 키가 0.516cm 커진다는 의미이다.
```

---

## **문제 2. 로지스틱 회귀**

> 🧚 다트비에서는 학생의 학업 성취도를 예측하기 위해 다항 로지스틱 회귀 분석을 수행하였습니다. 학업 성취도(Y)는 ‘낮음’, ‘보통’, ‘높음’ 3가지 범주로 구분되며, 독립 변수는 주당 공부 시간(Study Hours)과 출석률(Attendance Rate)입니다. 단, 기준범주는 '낮음' 입니다.
> 

| 변수 | Odds Ratio Estimates | 95% Wald Confidence Limits |
| --- | --- | --- |
| Study Hours | **2.34** | (1.89, 2.88) |
| Attendance Rate | **3.87** | (2.92, 5.13) |

> 🔍 Q1. Odds Ratio Estimates(오즈비, OR)의 의미를 해석하세요.
> 

<!--변수 Study Hours의 오즈비 값이 2.34라는 것과 Attendance Rate의 오즈비 값이 3.87이라는 것이 각각 무엇을 의미하는지 구체적으로 생각해보세요.-->

```
변수 study hours로 인해 높은 성취도를 가질 가능성이 낮은 성취도 대비 2.34배 더 크고, 변수 attendance rate로 인해 높은 성취도를 가질 가능성이 낮은 성취도 대비 3.87배 더 크다는 의미이다.
```

> 🔍 Q2. 95% Wald Confidence Limits의 의미를 설명하세요.
<!--각 변수의 신뢰구간에 제시된 수치가 의미하는 바를 생각해보세요.-->
> 

```
공부시간이 증가할수록 높은 학업 성취도를 가질 실제 효과 크기가, 낮은 성취도 대비 95%의 확률로 1.89에서 2.88사이에 있다.
출석률이 높을 수록 높은 학업 성취도를 가질 실제 효과 크기가, 낮은 성취도 대비 95%의 확률로 2.92에서 5.13 사이에 있다.
```

> 🔍 Q3. 이 분석을 기반으로 학업 성취도를 향상시키기 위한 전략을 제안하세요.
<!--Study Hours와 Attendance Rate 중 어느 변수가 학업 성취도에 더 큰 영향을 미치는지를 고려하여, 학업 성취도를 향상시키기 위한 효과적인 전략을 구체적으로 제시해보세요.-->
> 

```
attendance rate의 오즈비가 study hours의 오즈비보다 더 크기 때문에 공부 시간보다 학업 성취도에 더 큰 영향을 끼친다.
그러므로 출석률을 높이는 전략이 필요하다. 예를 들면 출석 점수 비중을 높이거나, 수업 시간에 참여가 필수적인 발표를 하는 수업 형태로 바꾼다.
```

---

## **문제 3. k-means 클러스터링**

> 🧚 선교는 고객을 유사한 그룹으로 분류하기 위해 k-means 클러스터링을 적용했습니다. 초기에는 3개의 군집으로 설정했지만, 결과가 만족스럽지 않았습니다. 선교가 최적의 군집 수를 찾기 위해 사용할 수 있는 방법을 한 가지 이상 제시하고 설명하세요.
> 

```
첫번째, 엘보우 기법이 있다. 이는 군집 내 중심점과 관측치 간 거리 합이 급감하는 구간의 k개수를 선정하는 방법이다.
두번째, 실루엣 계수이다. 군집 안의 관측치들이 다른 군집과 비교해서 얼마나 비슷한지 나타내는지 확인하여 k의 개수를 정하는 방법이다.
```

### 🎉 수고하셨습니다.
