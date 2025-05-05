# 통계학 4주차 정규과제

📌통계학 정규과제는 매주 정해진 분량의 『*데이터 분석가가 반드시 알아야 할 모든 것*』 을 읽고 학습하는 것입니다. 이번 주는 아래의 **Statistics_4th_TIL**에 나열된 분량을 읽고 `학습 목표`에 맞게 공부하시면 됩니다.

아래의 문제를 풀어보며 학습 내용을 점검하세요. 문제를 해결하는 과정에서 개념을 스스로 정리하고, 필요한 경우 추가자료와 교재를 다시 참고하여 보완하는 것이 좋습니다.

4주차는 `2부. 데이터 분석 준비하기`를 읽고 새롭게 배운 내용을 정리해주시면 됩니다.

## Statistics_4th_TIL

### 2부. 데이터 분석 준비하기

### 10. 데이터 탐색과 시각화

## Study Schedule

| 주차 | 공부 범위 | 완료 여부 |
| --- | --- | --- |
| 1주차 | 1부 p.2~56 | ✅ |
| 2주차 | 1부 p.57~79 | ✅ |
| 3주차 | 2부 p.82~120 | ✅ |
| 4주차 | 2부 p.121~202 | ✅ |
| 5주차 | 2부 p.203~254 | 🍽️ |
| 6주차 | 3부 p.300~356 | 🍽️ |
| 7주차 | 3부 p.357~615 | 🍽️ |

<!-- 여기까진 그대로 둬 주세요-->

# 10. 데이터 탐색과 시각화

```
✅ 학습 목표 :
* EDA의 목적을 설명할 수 있다.
* 주어진 데이터셋에서 이상치, 누락값, 분포 등을 식별하고 EDA 결과를 바탕으로 데이터셋의 특징을 해석할 수 있다.
* 공분산과 상관계수를 활용하여 두 변수 간의 관계를 해석할 수 있다.
* 적절한 시각화 기법을 선택하여 데이터의 특성을 효과적으로 전달할 수 있다.
```

- 탐색적 데이터 분석
- 상관성 분석
- 시각화 기법

> **Garbage In, Garbage Out**
> 

가치가 없는 잘못된 데이터를 사용하면 무가치한 결과가 나온다.

→ 다양한 각도에서 데이터를 탐색하고 시각화하여 **가치있는 데이터로 정제!**

**⚠ EDA와 데이터 시각화는 구별해야 함**

- EDA단계에서 효율적 데이터 파악을 위해 시각화를 하기도 함
- 시각화의 궁극적 목표는 분석 결과를 **커뮤니케이션하기 위함**

**📌 다양한 시각화의 방식**

- 시간 시각화
- 비교 시각화: 그룹별 차이를 나타내기 위함
- 분포 시각화: 특정 항목이 차지하는 비중을 나타내기 위함
- 관계 시각화: 두 개 이상의 수치 데이터를 통해 서로 간의 관계를 나타내기 위함
- 공간 시각화: 실제 지리적 위치에 수치를 나타냄

## 10.1 탐색적 데이터 분석

- 가공하지 않은 원천의 데이터를 있는 그대로 탐색하고 분석하는 기법
- 기술통계와 데이터 시각화를 통해 데이터 특성을 파악
- **극단적인 해석은 피해야 하며, 지나친 추론이나 자의적 해석도 지양해야 한다.**

### EDA를 하는 주요 목적

- 데이터의 형태와 척도가 분석에 알맞게 되어있는지 확인
- 데이터의 평균, 분산, 분포, 패턴 등의 확인을 통해 데이터 특성 파악
- 데이터의 결측값이나 이상치 파악 및 보완
- 변수 간의 관계성 파악
- 분석 목적과 방향성 점검 및 보정

![image](https://github.com/user-attachments/assets/096b1829-1db9-4624-86ef-a29f19d36061)


### **10.1.1 엑셀을 활용한 EDA**

- 가장 간단하면서 효과적인 방법
    - 각 데이터 샘플을 1000개씩 뽑아서 엑셀에 붙여 놓고 변수와 설명 리스트와 함께 눈으로 쭉 살펴보기
    - 실제로 해보면 도움이 된다 (ex: 성비, 가격대 형성, 판매와 도매 가격 차이…)

![image.png](attachment:edd19d4f-df95-446e-9d4e-f0d07650da9b:image.png)

- 김종혁 고객: 판매가격<도매가격
    - 데이터가 잘못 입력되었거나, 생각하지 못했던 할인과 같은 데이터 입력 프로세스 예상 가능
- 피벗 테이블 생성해서 성비, 평균 가격 등을 확인 가능
- 필요에 따라 간단한 그래프 그려서 데이터 파악 가능 ← 시각화가 EDA에 나오는 이유 !
- **엑셀이 제일 사용자 친화적이고 효율적인 프로그램이다**

### **10.1.2 탐색적 데이터 분석 실습**

```python
df.head()
```

![image.png](attachment:5b8e0edb-6cc4-4db9-a6b4-c78058ba9761:image.png)

- 데이터가 제대로 로드되었는지 확인

```python
df.info()
```

![image.png](attachment:6caf313c-dc80-45ea-b3ef-a78767f1f129:image.png)

- 데이터에 대한 전반적인 정보
- 행과 열의 크기와 각 칼럼을 구성하는 값의 자료형 등을 확인
- 숫자형이어야 하는 칼럼이 문자형이거나 문자형이어야 하는 칼럼이 숫자형으로 되어 있는 것은 없는지 확인
- 결측값 확인
    - children 칼럼의 경우 4건의 결측값: 표본 제거 방법 사용
    - company 칼럼은 90% 이상이 결측값: 특정 명으로 대치

```python
df.describe()
```

![image.png](attachment:e6921c4f-6b8b-421e-91dd-5a8ac7847998:image.png)

- 평균, 표준편차, 최대 최솟값 등을 한 번에 확인할 수 있다.
- arrival_date_year와 같이 숫자형이지만 문자형과 다름 없는 칼럼은 이러한 통계치가 큰 의미가 없다.
    - 각 빈도 등을 확인하는 방식으로 분석!

![image.png](attachment:c9efe1f6-564b-4a07-b3bf-0020d556b4a6:image.png)

- 왜도와 첨도 확인
- 정규성이 필요한 경우 로그변환, 정규화, 표준화 등의 방법을 사용

```python
df.kurtosis()
```

![image.png](attachment:06ec7d96-1904-43ec-81b8-e04a9b008dd7:image.png)

- 왜도: skew()
- 첨도: kurtosis
    - babies 칼럼처럼 분포가 넓지 않은 경우 값이 매우 높게 나옴

```python
sns.distplot(df["lead_time"])
```

![image.png](attachment:bddb0f8e-e59e-4d4d-9f98-80a3b6ea6f76:image.png)

- 칼럼의 분포를 시각화
- lead_time: 예약 날짜로부터 투숙 날짜까지의 일수 차이를 의미
- 0값이 확연히 많은 것 ⇒ 당일 체크인 하는 투숙객이 많은 편
    - 당일 예약인지, 시스템상 기록이 제대로 남지 않은 예약을 일괄적으로 0이라고 한 것인지 검토 필요

![image.png](attachment:faa9ab84-816d-46f3-a839-2d854f59a8fc:image.png)

- 호텔 구분에 따라 투숙객의 리드타이이 어떻게 다른지 시각화
- resort 호텔은 리드타임의 최댓값이 city호텔보다 높지만, 대체적으로 더 작은 값에 분포함

## 10.2 공분산과 상관성 분석

- 변수 간의 관계 파악
- 타깃 변수 Y와 입력 변수 X와의 관계, X들 간의 관계
- 다중 공선성 방지

![image.png](attachment:e4920759-7e01-4d9f-8da6-e8ee81cb50a0:image.png)

### **10.2.1 공분산**

- 서로 공유하는 분산
- 두 분산의 관계
- X1과 X2의 공통적인 분산의 정도를 알 수 있다
    - 0: 두 변수는 상관관계 없음
    - 양수: 양의 상관관계
    - 음수: 음의 상관관계

![image.png](attachment:d7a6a5c2-6686-4546-900f-81d553696edd:image.png)

**🎯예시**

![image.png](attachment:b80c1a21-8e72-4e95-bed6-a61b65f5491d:image.png)

- 웹사이트 접속 시간 편차: -24, 11, -4, 26, -9
- 각 고객의 구매 비용 편차: -29, 9, -8, 44, -16

![image.png](attachment:7a19134d-a8b8-4304-8606-071d86204a10:image.png)

### **10.2.2 상관계수**

- 공분산: 단순한 원리로 변수 간의 상관관계를 수치화
    - 각 변수 간의 다른 척도 기준이 그대로 반영되어 공분산 값이 지니는 크기가 상관성의 정도를 나타내지 못한다
    - 각 공분산을 비교하면서 상관관계가 더 크다고 말할 수 없다
- 공분산을 변수 각각의 표준편차 값으로 나누는 정규화를 통해 상관성 비교

⇒ 피어슨 상관계수

**피어슨 상관계수**

![image.png](attachment:bf41b078-966f-414e-954b-c392d4a5065e:image.png)

- -1<R<1

![image.png](attachment:f8bb1ee1-9e84-419e-995f-0c109d2d50dc:image.png)

![image.png](attachment:d8047162-1d5f-4db9-a128-70e58d216f23:image.png)

- 산점도의 기울기와 상관계수는 관련이 없다
    - 분산의 관계성이 같다면 기울기가 크든 작든 상관계수는 같다 !!
- 상관계수가 높다는 것은 설명력이 높다는 말
- 상관계수^2 = R^2= 결정계수: 총 변동 중에서 회귀선에 의해 설명되는 변동이 차지하는 비율
- 2차 방정식 그래프와 비슷한 모양이 될 경우 상관계수가 매우 낮게 측정될 수 있음
    - 상관계수가 0이 나오더라도 다른 통계적 관계나 패턴이 숨겨져 있을 수 있으니까 이것만 확인하면 안됨~~
    - 산점도도 같이 그려서 확인해보기

![image.png](attachment:fc34a5a1-0835-4ba2-9883-7e5c584b0387:image.png)

![image.png](attachment:54f8f78d-da09-4ec8-8636-037f04a38e9d:image.png)

### **10.2.3 공분산과 상관성 분석 실습**

```python
sns.set(font_scale=1.1)
sns.set_style("ticks")
sns.pairplot(df,
	diag_kind="kde"
	)
plt.show()
```

- 상관성을 확인할 수 있음

![image.png](attachment:5ebe3c34-c5be-489c-9d82-02dbed576514:image.png)

```python
df.cov()
```

- 공분산 확인할 때, free sulfur dioxide와 total sulfur dioxide변수가 높은 공분산을 보임

→ but, 공분산으로 변수 간 상관성을 분석하기에 가독성이 떨어짐

![image.png](attachment:7b810bae-d154-47dd-9b53-f2218102f80c:image.png)

```python
df.corr(method="person")
```

- 상관분석
- 알아서 문자형 변수 제외
- 고유번호같이 숫자형이지만 의미가 없는 경우에는 drop()함수로 제거해주기

![image.png](attachment:45cfb2d1-d92c-4bd1-8d90-4347a6439689:image.png)

```python
sns.heatmap(df.corr(),cmap="viridis")
```

![image.png](attachment:a5c96833-721e-4578-b4a1-46259bdbee5c:image.png)

- 노란색 - 양의 상관관계
- 보라색 - 음의 상관관계

```python
sns.clustermap(df.,corr(),
	annot=True,
	cmap="RdYlBu_r",
	vmin=-1, vmax=1,)
```

![image.png](attachment:a3a06a11-e126-4e1d-86c8-733d742b1371:image.png)

![image.png](attachment:a8ff786d-da46-4523-8e84-5af47ad79868:image.png)

- 중복 제거 히트맵 시각화
- 매트릭스의 우측 상단을 모두 true인 1로, 하단을 false인 0으로 변환하였다
- T/F mask 배열로 변환 후 히트맵 그래프 생성

## 10.3 시간 시각화

- 시계열 형태 → 데이터의 변화 표현
- 전체적인 흐름을 한눈에 알 수 있고, 데이터의 트렌드나 노이즈를 쉽게 찾아낼 수 있다
- 선 그래프
    - 시간 간격의 밀도가 높을 때 사용
    - 데이터 양이 많거나 변동이 심하면 트렌드나 패턴을 확인하기 어려움
    - → 추세선을 삽입하여서 안정된 선으로 표현
- 추체선을 그리는 방법 : 이동평균(moving average)
    - 연속적 그룹의 평균 구하기
    
    ![image.png](attachment:f74f1b94-2265-4c1a-a0e1-c3e09594718a:image.png)
    
- 분절형 시간 시각화
    - 막대 그래프, 누적 막대 그래프, 점 그래프
    - 월 간격 단위 흐름과 같이 **시간의 밀도가 낮은 경우에 활용하기 좋은 방법**
    - 값들의 상대적 차이를 나타내는 것에 유리
    - 누적 막대 그래프 - 한 시점에 2개 이상의 세부 항목이 존재할 때 사용

![image.png](attachment:4118a79d-144b-4fcb-9d7f-75ce38e585ad:image.png)

### 10.3.1 시간 시각화 실습

```python
df['Date2']=pd.to_datetime(df['Order Date'], infer_datetime_format=True)
df=df.sort_values(by='Date2')
df['Year']=df['Date2'].dt.year

df_line=df[df.Year==2018]

df_line=df_line.groupby('Date2')['Sales'].sum().reset_index()

df.line.head()
```

![image.png](attachment:cf568efb-1307-4bfe-970e-0a6275e0e3c5:image.png)

- date 칼럼 날짜 형식 변환 → 날짜 오름차순 정렬 → 연도 칼럼 생성
- 선그래프용 데이터셋을 생성한 뒤, 2018 데이터만 필터링 → 일별 매출액 가공

```python
df_line['Month']=df_line['Sales'].rolling(window=30).mean()

ax=df_line.plot(x='Date2', y='Sales',linewidth='0.5')
df_line.plot(x='Date2', y='Month',color='#FF7F50',linewidth='1', ax=ax)
```

![image.png](attachment:1584c086-b16a-49e6-8ccc-493c6bedbf16:image.png)

- 30일 이동평균 생성 → 선 그래프로 시각화
- rolling 함수를 통해 month 칼럼 새로 만들기 → plot 함수로 선그래프 생성

```python
df_bar_1=df.groupby('Year')['Sales'].sum().reset_index()

df_bar_1.head()
```

![image.png](attachment:5e135dc8-5031-4e7f-8b50-12ad256f4a00:image.png)

- year 칼럼으로 groupby()를 하여 연도별 매출액 합계를 만듦 → 2015년부터 2018년까지의 행이 생성

```python
ax=df_bar_1.plot.bar(x='Year', y='Sales', rot=0, figsize=(10,5))
```

![image.png](attachment:4b042e4c-f6f7-4bda-9366-fabfc287b104:image.png)

```python
df_bar_2=df.groupby(['Year','Segment'])['Sales'].sum().reset_index()

df_bar_2_pv=df_bar_2.pivot(index='Year', columns='Segment', values='Sales').reset_index()

df_bar_2_pv.head()
```

![image.png](attachment:d229ba9f-628b-4e28-98bc-b8a39c37fbef:image.png)

- 연도별, 고객 세그먼트별 그룹지어서 합계 계산
    - 세그먼트 칼럼을 활용하여 consumer, corporate, home office 구분에 따라 매출액 집계
- 고객 세그먼틀글 칼럼으로 피벗 테이블 형성

```python
df_bar_2_pv.plot.bar(x='Year', stacked=True, figsize=(10,7))
```

![image.png](attachment:46f553df-2e3d-4a12-9d4a-719b355ac04f:image.png)

## 10.4 비교 시각화

- 히트맵 차트: 그룹과 비교 요소가 많을 때 효과적으로 시각화를 할 수 있는 방법
    - 현재 가지고 있는 데이터의 구조와 자신이 확인하고자 하는 목적을 정확히 파악한 다음 차트를 그려야한다.
- 방사형 차트
    
    ![image.png](attachment:5f434733-aec8-4b6c-9b3b-b4f4d8a79d6d:image.png)
    
- 평행 좌표 그래프
    
    ![image.png](attachment:dada1d90-a717-4862-b5ed-025762d3c8cf:image.png)
    
    - 효과적이기 위해서 변수별 값을 정규화하면 됨

### 10.4.1 비교 시각화 실습

```python
df1=df[df['Tm'].isin(['ATL','BOS','BRK','CHI','CH0'])]

df1=df1[['Tm', 'ORB%', 'TRB%', 'AST%', 'BLK%', 'USG%']]

df1=df1.groupby('Tm').mean()
df1.head()
```

![image.png](attachment:18f18f87-744b-453c-b922-d1b6a114be59:image.png)

```python
fig=plt.figure(figsize=(8,8))
fig.set_facecolor('white')
plt.pcolor(df1.values)

plt.xticks(range(len(df1.columns)),df1.columns)
plt.yticks(range(len(df1.index)),df1.index)
plt.xlabel('Value', fontsize=14)
plt.ylabel('Team', fontsize=14)
plt.colorbar()
plt.show()
ax=fig.add_subplot(111, polar=True)
```

![image.png](attachment:cb41f096-7f94-4d4c-a189-0731bdb1e590:image.png)

- BRK의 어시스트 비율이 다른 팀에 비해 높음

```python
df2=df[df['Tm'].isin(['ATL','BOS','BRK','CHI','CHO'])]
df2=df2[['Tm','Age','G']]

df2=df2.groupby(['Tm','Age']).mean().reset_index()

df2=df2.pivot(index='Tm', columns='Age', values='G')

df2.head()
```

![image.png](attachment:36b61b8d-95f3-4ddf-910f-9f755e619d3f:image.png)

```python
fig=plt.figure(figsize=(8,8))
fig.set_facecolor('white')
plt.pcolor(df2.values)

plt.xticks(range(len(df2.columns)),df2.columns)
plt.yticks(range(len(df2.index)),df2.index)
plt.xlabel('Age', fontsize=14)
plt.ylabel('Team', fontsize=14)
plt.colorbar()
plt.show()
```

- BRK팀이 34세 선수의 게임참여 횟수가 약 30회 정도로 확연히 높다

![image.png](attachment:7b0afb06-9f53-4b9d-ac67-e7569ce26a5d:image.png)

```python
df3=df1.reset_index()
df3.head()
```

![image.png](attachment:14d72bc8-9ab0-452f-8656-992bb3898240:image.png)

```python
labels=df3.columns[1:]
num_labels=len(labels)

angles=[x/float(num_labels)*(2*pi) for x in range(num_labels)]
angles += angles[:1]

my_palette=plt.cm.get_cmap("Set2", len(df3.index))

fig=plt.figure(figsize=(12,15))
fig.set_facecolor('white')

for i, row in df3.iterrows():
    color = my_palette(i)
    data = df3.iloc[i].drop('Tm').tolist()
    data += data[:1]
    
    ax = plt.subplot(3,2,i+1, polar=True)
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1) 
    
    plt.xticks(angles[:-1], labels, fontsize=13)
    ax.tick_params(axis='x', which='major', pad=15)
    ax.set_rlabel_position(0)
    plt.yticks([0,5,10,15,20],['0','5','10','15','20'], fontsize=10) 
    plt.ylim(0,20)

    ax.plot(angles, data, color=color, linewidth=2, linestyle='solid')
    ax.fill(angles, data, color=color, alpha=0.4) 
    plt.title(row.Tm, size=16, color=color,x=0.5, y=1.1, ha='center') 
plt.tight_layout(pad=3) 
plt.show()
```

![image.png](attachment:d72cd92d-a5eb-474a-89df-36b18b4a709c:image.png)

- 깨짐..(사이즈가 안맞는듯)

```python

labels = df3.columns[1:]
num_labels = len(labels)

angles = [x/float(num_labels)*(2*pi) for x in range(num_labels)]
angles += angles[:1] 
    
my_palette = plt.cm.get_cmap("Set2", len(df3.index))
 
fig = plt.figure(figsize=(8,8))
fig.set_facecolor('white')
ax = fig.add_subplot(polar=True)
for i, row in df3.iterrows():
    color = my_palette(i)
    data = df3.iloc[i].drop('Tm').tolist()
    data += data[:1]
    
    ax.set_theta_offset(pi / 2) 
    ax.set_theta_direction(-1) 

    plt.xticks(angles[:-1], labels, fontsize=13)
    ax.tick_params(axis='x', which='major', pad=15) 
    ax.set_rlabel_position(0) 
    plt.yticks([0,5,10,15,20],['0','5','10','15','20'], fontsize=10) 
    plt.ylim(0,20)

    ax.plot(angles, data, color=color, linewidth=2, linestyle='solid', label=row.Tm) 
    ax.fill(angles, data, color=color, alpha=0.4) 
    
plt.legend(loc=(0.9,0.9))
plt.show()

```

- 이것도 깨짐
- 등분점과 시작점을 생성 → 시계방향으로 설정한 뒤에 각도 축 눈금, 여백 생성

```python
fig,axes=plt.subplots()
plt.figure(figsize=(16,8))
parallel_coordinates(df3,'Tm', ax=axes, colormap='winter',linewidth="0.5")
```

![image.png](attachment:49bf55bb-f53a-4cc8-9b55-68c3acae7ca6:image.png)

## 10.4 분포 시각화

- 연속형과 같은 양적 척도인지, 명목형과 같은 질적 척도인지에 따라 구분됨
- 구간 개수를 정하는 것이 중요
- 질적
    - 구성이 단순한 경우: 파이차트, 도넛 차트
    - 구성이 복잡한 경우: 트리맵 차트, 와플 차트(일정한 네모난 조각들로 분포 표현 but 위계 구조x)

### 10.5.1 분포 시각화 실습

```python
df1= df[['height_cm']]

plt.hist(df1, bins=10, label='bins=10')
plt.legend()
plt.show()
```

![image.png](attachment:b206053c-170a-44fc-bec8-a04ab135801b:image.png)

```python
df1_1=df[df['sex'].isin(['man'])]
df1_1=df1_1[['height_cm']]
df1_2=df[df['sex'].isin(['woman'])]
df1_2=df1_2[['height_cm']]

plt.hist(df1_1, color='green', alpha=0.2, bins=10, label='man', density=True)
plt.hist(df1_2, color='red', alpha=0.2, bins=10, label='woman', density=True)
plt.legend()
plt.show()
```

![image.png](attachment:6981fc4e-7d24-4e34-b291-316dcbbc4dcb:image.png)

```python
df2=df[['country','height_cm']]
df2=df2[df.height_cm>=175]
df2=df2.groupby('country').count().reset_index()
df2.head(10)
```

![image.png](attachment:f937a3ec-7ee5-4545-9d58-3d481aef63b1:image.png)

```python
fig = plt.figure(figsize=(8,8))
fig.set_facecolor('white') 
ax = fig.add_subplot()

ax.pie(df2.height_cm, 
       labels=df2.country,
       startangle=0, 
       counterclock=False, 
       autopct=lambda p : '{:.1f}%'.format(p) 
       )

plt.legend()
plt.show()
```

![image.png](attachment:292b713a-fb6c-4423-bcb1-6ca89056f0b7:image.png)

```python
wedgeprops={'width': 0.7, 'edgecolor': 'w', 'linewidth': 5}

plt.pie(df2.height_cm, labels=df2.country, autopct='%.1f%%', 
        startangle=90, counterclock=False, wedgeprops=wedgeprops)
plt.show()
```

![image.png](attachment:849c6f07-5271-4997-9397-195778af1222:image.png)

```python
df3 = df[['country', 'sex', 'height_cm']]
df3=df3[df.height_cm >= 175]

df3 = df3.groupby(['country','sex']).count().reset_index()

df3.head(10)
```

![image.png](attachment:734c1d42-ddc1-4bcf-87b3-06541bc551e8:image.png)

```python
fig = px.treemap(df3,
                 path=['sex','country'],
                 values='height_cm',
                 color='height_cm',
                 color_continuous_scale='viridis')

fig.show()
```

![image.png](attachment:aec7210d-c219-45f6-9a69-ef60c6ac3154:image.png)

```python
fig = plt.figure(
    FigureClass=Waffle,
    plots={
        111: {
            'values': df2['height_cm'],
            'labels': ["{0} ({1})".format(n, v) for n, v in df2['country'].items()],
            'legend': {'loc': 'upper left', 'bbox_to_anchor': (1.05, 1), 'fontsize': 8},
            'title': {'label': 'Waffle chart test', 'loc': 'left'}
        }
    },
    rows=10,
    figsize=(10, 10) 
)
```

![image.png](attachment:14f54472-5acc-410c-8a4c-76953a7eb8bf:image.png)

## 10.6 관계 시각화

- 산점도를 많이 이용하는데, 극단치를 제거하고 그리것이 중요함
    - 극단치때문에 주요 분포 구간이 압축되어 시각화의 효율이 떨어짐
    - 데이터가 많아서 점이 겹치면 투명도를 줘서 밀도 표현하기
- 버블차트
    - 3가지 요소의 상관관계 표현가능

### 10.6.1 관계 시각화 실습

```python
plt.scatter(df['R&D Spend'], df['Profit'], s=50, alpha=0.4)
plt.show()
```

![image.png](attachment:61b22d1d-ec23-4f60-a075-d26da5441e17:image.png)

```python
ax=sns.lmplot(x='R&D Spend', y='Profit', data=df)
```

![image.png](attachment:65646008-0d90-4245-8b94-b913bd86fcdc:image.png)

```python
plt.scatter(df['R&D Spend'], df['Profit'], s=df['Marketing Spend']*0.001, 
            c=df['Administration'], alpha=0.5, cmap='Spectral')
plt.colorbar()
plt.show()
```

![image.png](attachment:f60df3f5-3640-48a3-97d6-5fa3673e006c:image.png)

## 10.7 공간 시각화

- 분석 방향과 같이 스토리라인을 잡고 시각화를 적용하는 것이 좋다
- 도트맵
    - 지리적 위치에 동일한 크기의 작은 점ㅇ믈 찍어서 해당 지역의 데이터 분포나 패턴을 표현하는 기법
- 버블맵
    - 버블차트를 지도에 그대로 옮겨 둔 것
- 코로플레스맵
    - 단계 구분도, 데이터 값의 크기에 따라 색상의 음영을 달리하여 해당 지역에 대한 값을 시각화
    - 색상을 합치거나 투명도, 명도, 채도를 다양하게 표현 가능
- 커넥션맵/링크맵
    - 지도에 찍힌 점들을 곡선 또는 직선으로 연결하여 지리적 관계
    - 지도에 경로 표시!!

### 10.7.1 공간 시각화 실습

```python
m = folium.Map(location=[37.541, 126.986], zoom_start=12)
m
```

![image.png](attachment:1c32bd84-d313-4ac2-bfb0-446c291fbffd:image.png)

```python
m = folium.Map(location=[37.541, 126.986],tiles='Stamen Toner',zoom_start=12)

folium.CircleMarker([37.5538, 126.9810],radius=50, 
                    popup='Laurelhurst Park', color='#3246cc', 
                    fill_color='#3246cc').add_to(m)

folium.Marker([37.5538, 126.9810], popup='The Waterfront').add_to(m)
    
m
```

![image.png](attachment:406f5227-0a66-40b5-add6-327a7b6fae5a:image.png)

- 안됨 ㅠㅠ

```python
m = folium.Map([37.541, 126.986], zoom_start=12 ,width="100%", height="100%")
locations = list(zip(df.latitude, df.longitude))
cluster = plugins.MarkerCluster(locations=locations,                     
               popups=df["name"].tolist())  
m.add_child(cluster)
m
```

![image.png](attachment:1f06af71-206a-4e13-92c0-fe93107b34c1:image.png)

```python
df_m = df.groupby('gu_name').agg({'latitude':'mean',
                                  'longitude':'mean',
                                  'name':'count'}).reset_index()
df_m.head()
```

![image.png](attachment:c173a1c3-9edb-4961-8861-d3bd0e4f133c:image.png)

```python
m = folium.Map(location=[37.541, 126.986], tiles='Cartodb Positron', 
               zoom_start=11, width="100%", 
               height="100%")

folium.Choropleth(
    geo_data=geo, 
    fill_color="gray"
    ).add_to(m)

locations = list(zip(df_m.latitude, df_m.longitude))
for i in range(len(locations)):
    row = df_m.iloc[i]
    folium.CircleMarker(location=locations[i],
                        radius= float(row.name/2),
                        fill_color="blue"
                       ).add_to(m)
m
```

![image.png](attachment:39542d23-c327-4dea-a0da-d71f1781d3f4:image.png)

```python
import json

geo_path = "Seoul_Gu.json"

with open(geo_path, encoding='cp949') as f:
    geo = json.load(f)
```

```python
m = folium.Map([37.541, 126.986], zoom_start=12 ,width="100%", height="100%")
locations = list(zip(df.latitude, df.longitude))
cluster = plugins.MarkerCluster(locations=locations,                     
               popups=df["name"].tolist())  
m.add_child(cluster)
m
```

![image.png](attachment:9272a31e-f593-4ea8-a675-3da98b731005:image.png)

```python
m = folium.Map(location=[37.541, 126.986], zoom_start=12, width="100%", height="100%")
locations = list(zip(df.latitude, df.longitude))
for i in range(len(locations)):
    folium.CircleMarker(location=locations[i],radius=1).add_to(m)
m
```

![image.png](attachment:e73ccaaf-799e-4d85-82aa-a81ee3360b13:image.png)

```python
df_m = df.groupby('gu_name').agg({'latitude':'mean',
                                  'longitude':'mean',
                                  'name':'count'}).reset_index()
df_m.head()
```

![image.png](attachment:6cd50cb9-fb63-4a38-b0b8-60b87804b246:image.png)

```python
m = folium.Map(location=[37.541, 126.986], tiles='Cartodb Positron', 
               zoom_start=11, width="100%", 
               height="100%")

folium.Choropleth(
    geo_data=geo, 
    fill_color="gray"
    ).add_to(m)

locations = list(zip(df_m.latitude, df_m.longitude))
for i in range(len(locations)):
    row = df_m.iloc[i]
    folium.CircleMarker(location=locations[i],
                        radius= float(row.name/2),
                        fill_color="blue"
                       ).add_to(m)
m
```

![image.png](attachment:1c7d8d0c-9b49-466f-b0d9-a82c85e17463:image.png)

```python
source_to_dest = zip([37.541,37.541,37.541,37.541,37.541], 
                     [35.6804, 38.9072, 14.5995, 48.8566,55.7558],
                     [126.986,126.986,126.986,126.986,126.986], 
                     [139.7690, -77.0369, 120.9842, 2.3522,37.6173])

fig = go.Figure()

for a, b, c, d in source_to_dest:
    fig.add_trace(go.Scattergeo(
                        lat = [a, b],
                        lon = [c, d],
                        mode = 'lines',
                        line = dict(width = 1, color="red"),
                        opacity = 0.5
                        ))

fig.update_layout(
                margin={"t":0,"b":0,"l":0, "r":0, "pad":0},
                showlegend=False,
                geo = dict(
                showcountries=True)
                )

fig.show()
```

![image.png](attachment:5bfa0b1c-43f1-421a-8fa0-16559a22a656:image.png)

## 10.8 박스플롯

- 하나의 그림으로 양적 척도 데이터의 분포 및 편향성, 평균과 중앙값 등 다양한 수치를 보기 쉽게 정리!
- 두 변수의 값을 비교할 때 효과적
- 박스 플롯의 수치
    - 최솟값
    - 제1사분위
    - 제2사분위
    - 제3사분위
    - 최댓값

![image.png](attachment:f1359a7a-e68c-49e2-ac85-e216f019f4fb:image.png)

![image.png](attachment:ef8a2f20-9360-4d35-ace4-02a3c5d461cf:image.png)

- 항상 데이터 분포도를 함께 떠올리는 습관이 필요함!!

### 10.8.1 박스 플롯 실습

```python
plt.figure(figsize = (8, 6))
sns.boxplot(y = 'Profit', data = df)
plt.show()

plt.figure(figsize = (8, 2))
sns.boxplot(x = 'Profit', data = df)
plt.show()
```

![image.png](attachment:1e630b48-1f83-4263-82a6-a8b2c49fce67:image.png)

![image.png](attachment:03a16c8b-30d3-4372-90fc-1d0a7effffff:image.png)

```python
plt.figure(figsize=(8,5))
sns.boxplot(x="State", y="Profit", data=df)
plt.show()
```

![image.png](attachment:b660acf1-d710-488b-842a-2c3776acc21c:image.png)

```python
sns.boxplot(x="State", y="Profit", 
            showmeans=True, 
            boxprops={'facecolor':'None'}, 
            data=df)

sns.stripplot(x='State', y='Profit', 
              data=df, 
              jitter=True, 
              marker='o', 
              alpha=0.5,
              color='black')

plt.show()
```

![image.png](attachment:add56350-f436-4ec7-babf-640530d97f67:image.png)

# 확인 문제

## 문제 1.

> 🧚 공분산과 상관계수의 차이점에 대해 간단히 설명하세요.
> 

```
공분산은 단순히 각 변수의 편차를 곱한 값들을 n-1로 나눈 값으로 상관관계를 수치화시킨 값이다. 그렇기 때문에 각 변수 간의 다른 척도 기준이 적용된다.  
상관계수는 공분산을 각각의 표준편차 값으로 나누는 정규화를 한 값으로 피어슨 상관계수를 많이 사용한다.
```

## 문제 2.

> 🧚 다음 데이터 분석 목표에 적합한 시각화 방법을 보기에서 모두 골라 연결해주세요.
> 

> 보기: 산점도, 선그래프, 막대그래프, 히스토그램, 박스플롯
> 

(a) 변수의 분포 확인

(b) 두 변수 간의 관계 확인

(c) 집단별 평균 비교

(d) 시계열 데이터 분석

<!--중복 가능-->

```
(a) 변수의 분포 확인: 히스토그램, 박스 플롯

(b) 두 변수 간의 관계 확인: 산점도

(c) 집단별 평균 비교: 선그래프, 박스 플롯

(d) 시계열 데이터 분석: 선그래프, 막대 그래프
```

### 🎉 수고하셨습니다.
