# 통계학 4주차 정규과제

📌통계학 정규과제는 매주 정해진 분량의 『*데이터 분석가가 반드시 알아야 할 모든 것*』 을 읽고 학습하는 것입니다. 이번 주는 아래의 **Statistics_4th_TIL**에 나열된 분량을 읽고 `학습 목표`에 맞게 공부하시면 됩니다.

아래의 문제를 풀어보며 학습 내용을 점검하세요. 문제를 해결하는 과정에서 개념을 스스로 정리하고, 필요한 경우 추가자료와 교재를 다시 참고하여 보완하는 것이 좋습니다.

4주차는 `2부. 데이터 분석 준비하기`를 읽고 새롭게 배운 내용을 정리해주시면 됩니다.

---
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
---
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

![image](https://github.com/user-attachments/assets/e932578d-b131-4c3d-849a-62de453b4533)


- 김종혁 고객: 판매가격<도매가격
    - 데이터가 잘못 입력되었거나, 생각하지 못했던 할인과 같은 데이터 입력 프로세스 예상 가능
- 피벗 테이블 생성해서 성비, 평균 가격 등을 확인 가능
- 필요에 따라 간단한 그래프 그려서 데이터 파악 가능 ← 시각화가 EDA에 나오는 이유 !
- **엑셀이 제일 사용자 친화적이고 효율적인 프로그램이다**

### **10.1.2 탐색적 데이터 분석 실습**

```python
df.head()
```

![image](https://github.com/user-attachments/assets/770da2ce-bbe5-449a-9777-040a373d74c7)

- 데이터가 제대로 로드되었는지 확인

```python
df.info()
```

![image](https://github.com/user-attachments/assets/fc94facf-266a-4b03-bda9-10ba4f9e11af)

- 데이터에 대한 전반적인 정보
- 행과 열의 크기와 각 칼럼을 구성하는 값의 자료형 등을 확인
- 숫자형이어야 하는 칼럼이 문자형이거나 문자형이어야 하는 칼럼이 숫자형으로 되어 있는 것은 없는지 확인
- 결측값 확인
    - children 칼럼의 경우 4건의 결측값: 표본 제거 방법 사용
    - company 칼럼은 90% 이상이 결측값: 특정 명으로 대치

```python
df.describe()
```

![image](https://github.com/user-attachments/assets/0d6cf468-af95-48b1-ab67-63b913864f4e)

- 평균, 표준편차, 최대 최솟값 등을 한 번에 확인할 수 있다.
- arrival_date_year와 같이 숫자형이지만 문자형과 다름 없는 칼럼은 이러한 통계치가 큰 의미가 없다.
    - 각 빈도 등을 확인하는 방식으로 분석!

![image](https://github.com/user-attachments/assets/55be443a-51d9-4c60-9650-8db7a5e17822)


- 왜도와 첨도 확인
- 정규성이 필요한 경우 로그변환, 정규화, 표준화 등의 방법을 사용

```python
df.kurtosis()
```

![image](https://github.com/user-attachments/assets/da370a20-73f3-4ce6-9542-3673ea61c795)

- 왜도: skew()
- 첨도: kurtosis
    - babies 칼럼처럼 분포가 넓지 않은 경우 값이 매우 높게 나옴

```python
sns.distplot(df["lead_time"])
```

![image](https://github.com/user-attachments/assets/9be6a3a8-7f4b-4d24-97be-401d65f95ec3)


- 칼럼의 분포를 시각화
- lead_time: 예약 날짜로부터 투숙 날짜까지의 일수 차이를 의미
- 0값이 확연히 많은 것 ⇒ 당일 체크인 하는 투숙객이 많은 편
    - 당일 예약인지, 시스템상 기록이 제대로 남지 않은 예약을 일괄적으로 0이라고 한 것인지 검토 필요

![image](https://github.com/user-attachments/assets/776b3567-9f0c-4256-99dc-626481285eeb)

- 호텔 구분에 따라 투숙객의 리드타이이 어떻게 다른지 시각화
- resort 호텔은 리드타임의 최댓값이 city호텔보다 높지만, 대체적으로 더 작은 값에 분포함

## 10.2 공분산과 상관성 분석

- 변수 간의 관계 파악
- 타깃 변수 Y와 입력 변수 X와의 관계, X들 간의 관계
- 다중 공선성 방지

![image](https://github.com/user-attachments/assets/86d6e69b-6c9d-4218-83a2-301589352e70)


### **10.2.1 공분산**

- 서로 공유하는 분산
- 두 분산의 관계
- X1과 X2의 공통적인 분산의 정도를 알 수 있다
    - 0: 두 변수는 상관관계 없음
    - 양수: 양의 상관관계
    - 음수: 음의 상관관계

![image](https://github.com/user-attachments/assets/6f799136-b316-480f-ac7d-11faa75160a4)

**🎯예시**

![image](https://github.com/user-attachments/assets/1de5bd55-aae8-4a7a-9c9e-278e5c46f102)


- 웹사이트 접속 시간 편차: -24, 11, -4, 26, -9
- 각 고객의 구매 비용 편차: -29, 9, -8, 44, -16

![image](https://github.com/user-attachments/assets/ea8d0a1b-86cf-4d7f-9257-ab9660697bbd)


### **10.2.2 상관계수**

- 공분산: 단순한 원리로 변수 간의 상관관계를 수치화
    - 각 변수 간의 다른 척도 기준이 그대로 반영되어 공분산 값이 지니는 크기가 상관성의 정도를 나타내지 못한다
    - 각 공분산을 비교하면서 상관관계가 더 크다고 말할 수 없다
- 공분산을 변수 각각의 표준편차 값으로 나누는 정규화를 통해 상관성 비교

⇒ 피어슨 상관계수

**피어슨 상관계수**

![image](https://github.com/user-attachments/assets/eef9504c-e7c4-46c6-8b4f-b582999af717)


- -1<R<1

![image](https://github.com/user-attachments/assets/046fd1eb-859f-4fb8-a421-cd8766a4e244)
![image](https://github.com/user-attachments/assets/2ed48f8e-6654-42b5-b358-14c991086b41)


- 산점도의 기울기와 상관계수는 관련이 없다
    - 분산의 관계성이 같다면 기울기가 크든 작든 상관계수는 같다 !!
- 상관계수가 높다는 것은 설명력이 높다는 말
- 상관계수^2 = R^2= 결정계수: 총 변동 중에서 회귀선에 의해 설명되는 변동이 차지하는 비율
- 2차 방정식 그래프와 비슷한 모양이 될 경우 상관계수가 매우 낮게 측정될 수 있음
    - 상관계수가 0이 나오더라도 다른 통계적 관계나 패턴이 숨겨져 있을 수 있으니까 이것만 확인하면 안됨~~
    - 산점도도 같이 그려서 확인해보기

![image](https://github.com/user-attachments/assets/be2e8145-6c0a-4701-a534-8e091c13a185)
![image](https://github.com/user-attachments/assets/f7d73f67-62c3-4d0e-853d-654c3ca49bd6)

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

![image](https://github.com/user-attachments/assets/f78d077c-6a09-4083-928d-5f469cd09447)


```python
df.cov()
```

- 공분산 확인할 때, free sulfur dioxide와 total sulfur dioxide변수가 높은 공분산을 보임

→ but, 공분산으로 변수 간 상관성을 분석하기에 가독성이 떨어짐

![image](https://github.com/user-attachments/assets/f5d861f0-2793-4a04-aba0-e20df0af2e0f)


```python
df.corr(method="person")
```

- 상관분석
- 알아서 문자형 변수 제외
- 고유번호같이 숫자형이지만 의미가 없는 경우에는 drop()함수로 제거해주기

![image](https://github.com/user-attachments/assets/374aa20d-7451-4b3f-b42d-6bf50c6de5d2)

```python
sns.heatmap(df.corr(),cmap="viridis")
```

![image](https://github.com/user-attachments/assets/45cdb031-1b14-40a2-bc62-703a8f09ab02)

- 노란색 - 양의 상관관계
- 보라색 - 음의 상관관계

```python
sns.clustermap(df.,corr(),
	annot=True,
	cmap="RdYlBu_r",
	vmin=-1, vmax=1,)
```

![image](https://github.com/user-attachments/assets/f66faa0c-2460-46b1-bacc-553a6aa7043e)

![image](https://github.com/user-attachments/assets/fb637230-67a8-4630-8134-a3f1b61151b7)


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
    
   ![image](https://github.com/user-attachments/assets/67696b23-486c-4f37-819b-de5aad1851c4)


    
- 분절형 시간 시각화
    - 막대 그래프, 누적 막대 그래프, 점 그래프
    - 월 간격 단위 흐름과 같이 **시간의 밀도가 낮은 경우에 활용하기 좋은 방법**
    - 값들의 상대적 차이를 나타내는 것에 유리
    - 누적 막대 그래프 - 한 시점에 2개 이상의 세부 항목이 존재할 때 사용

![image](https://github.com/user-attachments/assets/d8eb9863-8a48-4210-9c4a-68b2fc03aa68)



### 10.3.1 시간 시각화 실습

```python
df['Date2']=pd.to_datetime(df['Order Date'], infer_datetime_format=True)
df=df.sort_values(by='Date2')
df['Year']=df['Date2'].dt.year

df_line=df[df.Year==2018]

df_line=df_line.groupby('Date2')['Sales'].sum().reset_index()

df.line.head()
```

![image](https://github.com/user-attachments/assets/82601852-9136-47de-8b02-506431aee293)

- date 칼럼 날짜 형식 변환 → 날짜 오름차순 정렬 → 연도 칼럼 생성
- 선그래프용 데이터셋을 생성한 뒤, 2018 데이터만 필터링 → 일별 매출액 가공

```python
df_line['Month']=df_line['Sales'].rolling(window=30).mean()

ax=df_line.plot(x='Date2', y='Sales',linewidth='0.5')
df_line.plot(x='Date2', y='Month',color='#FF7F50',linewidth='1', ax=ax)
```

![image](https://github.com/user-attachments/assets/a3c96b98-44be-4f80-b9a1-7b9fd75057c9)


- 30일 이동평균 생성 → 선 그래프로 시각화
- rolling 함수를 통해 month 칼럼 새로 만들기 → plot 함수로 선그래프 생성

```python
df_bar_1=df.groupby('Year')['Sales'].sum().reset_index()

df_bar_1.head()
```

![image](https://github.com/user-attachments/assets/1db8fffe-bc81-4b2e-9eae-cfacbee7f741)


- year 칼럼으로 groupby()를 하여 연도별 매출액 합계를 만듦 → 2015년부터 2018년까지의 행이 생성

```python
ax=df_bar_1.plot.bar(x='Year', y='Sales', rot=0, figsize=(10,5))
```

![image](https://github.com/user-attachments/assets/0dcf1f33-01ef-4334-adce-01c65df41d84)


```python
df_bar_2=df.groupby(['Year','Segment'])['Sales'].sum().reset_index()

df_bar_2_pv=df_bar_2.pivot(index='Year', columns='Segment', values='Sales').reset_index()

df_bar_2_pv.head()
```

![image](https://github.com/user-attachments/assets/49e07c59-6871-40b5-a2de-e48409bcd8da)


- 연도별, 고객 세그먼트별 그룹지어서 합계 계산
    - 세그먼트 칼럼을 활용하여 consumer, corporate, home office 구분에 따라 매출액 집계
- 고객 세그먼틀글 칼럼으로 피벗 테이블 형성

```python
df_bar_2_pv.plot.bar(x='Year', stacked=True, figsize=(10,7))
```

![image](https://github.com/user-attachments/assets/57bba985-c27d-43bc-9630-e06c52871fcc)


## 10.4 비교 시각화

- 히트맵 차트: 그룹과 비교 요소가 많을 때 효과적으로 시각화를 할 수 있는 방법
    - 현재 가지고 있는 데이터의 구조와 자신이 확인하고자 하는 목적을 정확히 파악한 다음 차트를 그려야한다.
- 방사형 차트
    
    ![image](https://github.com/user-attachments/assets/d7d9c781-5521-4651-ac49-8380930bd646)

    
- 평행 좌표 그래프
    
   ![image](https://github.com/user-attachments/assets/c6810ac8-5ca0-4faf-bb08-6bcf0251746b)

    
    - 효과적이기 위해서 변수별 값을 정규화하면 됨

### 10.4.1 비교 시각화 실습

```python
df1=df[df['Tm'].isin(['ATL','BOS','BRK','CHI','CH0'])]

df1=df1[['Tm', 'ORB%', 'TRB%', 'AST%', 'BLK%', 'USG%']]

df1=df1.groupby('Tm').mean()
df1.head()
```

![image](https://github.com/user-attachments/assets/ca635ece-a910-4fd2-b316-d527ac982d14)

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

![image](https://github.com/user-attachments/assets/c7fe8314-31ce-4e39-b6b3-f2c8e3d3fc3c)


- BRK의 어시스트 비율이 다른 팀에 비해 높음

```python
df2=df[df['Tm'].isin(['ATL','BOS','BRK','CHI','CHO'])]
df2=df2[['Tm','Age','G']]

df2=df2.groupby(['Tm','Age']).mean().reset_index()

df2=df2.pivot(index='Tm', columns='Age', values='G')

df2.head()
```

![image](https://github.com/user-attachments/assets/2b7d4f27-1d2d-47b9-aef4-a5e1e52e918a)


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

![image](https://github.com/user-attachments/assets/17af3d69-da06-4f34-9bf8-e32a8b265e49)


```python
df3=df1.reset_index()
df3.head()
```

![image](https://github.com/user-attachments/assets/cd856d32-205f-4764-9bb5-47013a3a1e94)


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

![image](https://github.com/user-attachments/assets/a384b1a0-39f0-49be-b4a7-8dcc6f94a884)


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

![image](https://github.com/user-attachments/assets/c2755202-6716-4d5d-804f-cbf9eb321485)


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

![image](https://github.com/user-attachments/assets/455171fe-6cb9-416e-ad3a-45ba9ee0a914)

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

![image](https://github.com/user-attachments/assets/cb63c321-8fac-4702-afab-da821efa9cc2)


```python
df2=df[['country','height_cm']]
df2=df2[df.height_cm>=175]
df2=df2.groupby('country').count().reset_index()
df2.head(10)
```

![image](https://github.com/user-attachments/assets/39e3b15f-2f76-4973-83dc-81ae188704f6)


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

![image](https://github.com/user-attachments/assets/1e5f63b9-ef15-4d6b-8c50-c35735968dc4)

```python
wedgeprops={'width': 0.7, 'edgecolor': 'w', 'linewidth': 5}

plt.pie(df2.height_cm, labels=df2.country, autopct='%.1f%%', 
        startangle=90, counterclock=False, wedgeprops=wedgeprops)
plt.show()
```

![image](https://github.com/user-attachments/assets/c446b38b-ae64-4c39-b060-19e1216cadd3)


```python
df3 = df[['country', 'sex', 'height_cm']]
df3=df3[df.height_cm >= 175]

df3 = df3.groupby(['country','sex']).count().reset_index()

df3.head(10)
```

![image](https://github.com/user-attachments/assets/fef13a18-051a-4365-bd4c-c24390268209)


```python
fig = px.treemap(df3,
                 path=['sex','country'],
                 values='height_cm',
                 color='height_cm',
                 color_continuous_scale='viridis')

fig.show()
```

![image](https://github.com/user-attachments/assets/19e10760-f500-4922-a886-30453a890477)


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

![image](https://github.com/user-attachments/assets/2ae8b670-b4ad-4e40-bed7-99b783ff3160)


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

![image](https://github.com/user-attachments/assets/ea09c524-b8a7-4093-9fb7-bcc0d15668bd)


```python
ax=sns.lmplot(x='R&D Spend', y='Profit', data=df)
```

![image](https://github.com/user-attachments/assets/4edf8740-6d31-4ff8-b4c7-418b303916f6)


```python
plt.scatter(df['R&D Spend'], df['Profit'], s=df['Marketing Spend']*0.001, 
            c=df['Administration'], alpha=0.5, cmap='Spectral')
plt.colorbar()
plt.show()
```

![image](https://github.com/user-attachments/assets/f680dfb9-1e37-4de5-8649-889f89046e84)


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

![image](https://github.com/user-attachments/assets/56e2fc9f-a996-4c45-833e-021e386126d4)


```python
m = folium.Map(location=[37.541, 126.986],tiles='Stamen Toner',zoom_start=12)

folium.CircleMarker([37.5538, 126.9810],radius=50, 
                    popup='Laurelhurst Park', color='#3246cc', 
                    fill_color='#3246cc').add_to(m)

folium.Marker([37.5538, 126.9810], popup='The Waterfront').add_to(m)
    
m
```

![image](https://github.com/user-attachments/assets/1b5beaf8-f89b-4a0b-8f00-f0804cf39725)
- 안됨 ㅠㅠ

```python
m = folium.Map([37.541, 126.986], zoom_start=12 ,width="100%", height="100%")
locations = list(zip(df.latitude, df.longitude))
cluster = plugins.MarkerCluster(locations=locations,                     
               popups=df["name"].tolist())  
m.add_child(cluster)
m
```

![image](https://github.com/user-attachments/assets/92b58076-267f-4f14-b170-ea552c4eb65d)

```python
df_m = df.groupby('gu_name').agg({'latitude':'mean',
                                  'longitude':'mean',
                                  'name':'count'}).reset_index()
df_m.head()
```

![image](https://github.com/user-attachments/assets/6ade8d2f-9dd1-424a-8243-13c810fe1cb4)


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

![image](https://github.com/user-attachments/assets/ea0e4301-8cf4-4ab3-b046-5b2dd2f0b700)

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

![image](https://github.com/user-attachments/assets/69e0c250-5fae-4fcf-8031-391838aec46c)


```python
m = folium.Map(location=[37.541, 126.986], zoom_start=12, width="100%", height="100%")
locations = list(zip(df.latitude, df.longitude))
for i in range(len(locations)):
    folium.CircleMarker(location=locations[i],radius=1).add_to(m)
m
```

![image](https://github.com/user-attachments/assets/fcb9e80d-e2e1-423b-bd6f-6356bd564708)

```python
df_m = df.groupby('gu_name').agg({'latitude':'mean',
                                  'longitude':'mean',
                                  'name':'count'}).reset_index()
df_m.head()
```

![image](https://github.com/user-attachments/assets/ea6a72fc-ee0f-4a83-8646-c240d8a5ac47)


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

![image](https://github.com/user-attachments/assets/326294f8-1e15-491a-b89d-640a0615826f)

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

![image](https://github.com/user-attachments/assets/c3ea8db9-694a-44a8-90b9-28f907a17d1b)

## 10.8 박스플롯

- 하나의 그림으로 양적 척도 데이터의 분포 및 편향성, 평균과 중앙값 등 다양한 수치를 보기 쉽게 정리!
- 두 변수의 값을 비교할 때 효과적
- 박스 플롯의 수치
    - 최솟값
    - 제1사분위
    - 제2사분위
    - 제3사분위
    - 최댓값

![image](https://github.com/user-attachments/assets/77f9bce0-80fe-47d6-96d0-67459c0740db)

![image](https://github.com/user-attachments/assets/2b52336f-3b68-4b3b-95d4-2d0e2c1bc574)


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

![image](https://github.com/user-attachments/assets/e68dd404-20f6-4806-9cc8-90e096199a55)
![image](https://github.com/user-attachments/assets/ff6baec3-c5ca-4e20-ad27-3aa716f03641)


```python
plt.figure(figsize=(8,5))
sns.boxplot(x="State", y="Profit", data=df)
plt.show()
```

![image](https://github.com/user-attachments/assets/cbeb334b-1323-4b80-b046-123b3606d893)

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

![image](https://github.com/user-attachments/assets/5099d596-b296-4399-897f-8e9520bc0078)


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
