# 통계학 3주차 정규과제

📌통계학 정규과제는 매주 정해진 분량의 『*데이터 분석가가 반드시 알아야 할 모든 것*』 을 읽고 학습하는 것입니다. 이번 주는 아래의 **Statistics_3rd_TIL**에 나열된 분량을 읽고 `학습 목표`에 맞게 공부하시면 됩니다.

아래의 문제를 풀어보며 학습 내용을 점검하세요. 문제를 해결하는 과정에서 개념을 스스로 정리하고, 필요한 경우 추가자료와 교재를 다시 참고하여 보완하는 것이 좋습니다.

2주차는 `2부-데이터 분석 준비하기`를 읽고 새롭게 배운 내용을 정리해주시면 됩니다.

## Statistics_3rd_TIL

### 2부. 데이터 분석 준비하기

### 08. 분석 프로젝트 준비 및 기획

### 09. 분석 환경 세팅하기

## Study Schedule

| 주차 | 공부 범위 | 완료 여부 |
| --- | --- | --- |
| 1주차 | 1부 p.2~56 | ✅ |
| 2주차 | 1부 p.57~79 | ✅ |
| 3주차 | 2부 p.82~120 | ✅ |
| 4주차 | 2부 p.121~202 | 🍽️ |
| 5주차 | 2부 p.203~254 | 🍽️ |
| 6주차 | 3부 p.300~356 | 🍽️ |
| 7주차 | 3부 p.357~615 | 🍽️ |

<!-- 여기까진 그대로 둬 주세요-->

# 08. 분석 프로젝트 준비 및 기획

```
✅ 학습 목표 :
* 데이터 분석 프로세스를 설명할 수 있다.
* 비즈니스 문제를 정의할 때 주의할 점을 설명할 수 있다.
* 외부 데이터를 수집하는 방법에 대해 인식한다.

```

## 8.1 데이터 분석의 전체 프로세스

> **8.1.1 데이터 분석의 3단계**
> 
- 효과적인 결정을 할 수 있도록 도움을 주는 것이 데이터 분석의 주된 목적

![image](https://github.com/user-attachments/assets/9c6967f1-002a-419a-8b2d-61ec01c05df0)


**(1) 설계 단계**

- 무엇을 하고자 하는지를 정확히 정의하고 프로젝트를 수행할 인력을 구성
- 실시간으로 피드백을 받을 수 있는 접촉 체계를 세우는 것이 좋으며, 적어도 일주일에 한 번 이상은 정기적인 미팅을 통해 진행 상황을 공유하는 것이 좋다

**(2) 분석 및 모델링 단계**

- 데이터 분석 및 모델링을 위한 서버 환경을 마련하고 본격적인 데이터 분석과 모델링
    - 데이터 분석부터 모델 검증 및 적용
- 데이터 추출, 검토, 가공, 모델링 등의 세부 절차와 부분 반복이 필요
- 모델의 비즈니스적 적합성을 심도있게 분석하고 성능을 평가하는 것이 중요
- 예시
    - KDD 분석 방법론, CRISP-DM 방법론, SAS사의 SEMMA 방법론
    - CRISP-DM과 SEMMA 중심으로 소개

**(3) 구축 및 활용 단계**

- 최종적으로 선정된 분석 모델을 실제 업무에 적용하고 그 성과 측정
- 때에 따라서는 분석 모델을 적용하기 위한 IT 시스템 구축 필요
- 모델이 적용된 후에는 기존보다 얼마나 개선됐는지 효과를 측정하고 평가
    - A/B 테스트를 통해 모델 성과 측정

> **8.1.2 CRISP-DM 방법론**
> 

![image](https://github.com/user-attachments/assets/3a5d77a1-5335-4ffb-848e-58473a44ccdd)


**1단계 - 비즈니스 이해**

---

- 현재 상황 평가
- 데이터 마이닝 목표 결정
- 프로젝트 계획 수립

**2단계 - 데이터 이해**

---

- 데이터 설명
- 데이터 탐색
- 데이터 품질 확인

**3단계 - 데이터 준비**

---

- 데이터 선택
- 데이터 정체
- 필수 데이터 구성
- 데이터 통합

**4단계 - 모델링**

---

- 모델링 기법 선정
- 테스트 디자인 생성
- 모델 생성
- 모델 평가

**5단계 - 평가**

---

- 결과 평가
- 프로세스 검토
- 다음 단계 결정

**6단계 - 배포**

---

- 배포 계획
- 모니터링 및 유지 관리 계획
- 최종 보고서 작성
- 프로젝트 검토

> **8.1.3 SAS SEMMA 방법론**
> 

![image](https://github.com/user-attachments/assets/9c8b6753-8cb2-46a9-abce-556ffe508dca)

**Sampling 단계**

---

- 전체 데이터에서 분석용 데이터 추출
- 의미 있는 정보를 추출하기 위한 데이터 분할 및 병합
- 표본추출을 통해 대표성을 가진 분석용 데이터 생성
- 분석 모델 생성을 위한 학습, 검증, 테스트 데이터셋 분할

**Exploration 단계**

---

- 통계치 확인, 그래프 생성 등을 통해 데이터 탐색
- 상관분석, 클러스터링 등을 통해 변수 간의 관계 파악
- 분석 모델에 적합한 변수 선정
- 데이터 현황을 파악하여 비즈니스 아이디어 도출 및 분석 방향 수정

**Modification 단계**

---

- 결측값 처리 및 최종 분석 변수 선정
- 로그변환, 구간화 등 데이터 가공
- 주성분분석(PCA) 등을 통해 새로운 변수 생성

**Modeling 단계**

---

- 다양한 데이터마이닝 기법 적용에 대한 적합성 검토
- 비즈니스 목적에 맞는 분석 모델을 선정하여 분석 알고리즘 적용
- 지도학습, 비지도학습, 강화학습 등 데이터형태에 따라 알맞은 모델 선정
- 분석 환경 인프라 성능과 모델 정확도를 고려한 모델 세부 옵션 설정

**Assessment 단계**

---

- 구축한 모델들의 예측력 등 성능을 비교, 분석, 평가
- 비즈니스 상황에 맞는 적정 임계치 설정
- 분석 모델 결과를 비즈니스 인사이트에 적용
- 상황에 따라 추가적인 데이터 분석 수행

![image](https://github.com/user-attachments/assets/eb76650c-9710-4c4c-9ad0-dcb7eb237d2b)

- **초반부:** 비즈니스 문제와 해결 방향을 명확히 정의하고 데이터를 탐색
- 중반부: 데이터를 목적에 맞도록 수집 및 가공하고 필요에 따라 머신러닝 모델을 사용
- 후반부: 데이터 분석 결과를 검토 및 검증하고 실제 환경에 적용
- 적용한 방법의 효과를 지속적으로 모니터링하고 성과 측정, 보완

## 8.2 비즈니스 문제 정의와 분석 목적 도출

- 성공적인 데이터 분석 프로젝트를 위해서는 프로젝트를 시작하기 전에 *현재의 문제를 명확하게 정의하고, 그에 맞는 데이터 분석 목적을 설정해야 한다.*
- 제대로 된 성과를 얻지 못하는 이유는 미디어에 퍼진 빅데이터 활용 성공사례들을 본 기업이나 집단에서 *불안한 마음에 별다른 사전 준비 없이 무작정 데이터 분석을 하려고 하기 때문이다.*
- 채찍효과: 공급사슬에서 수요 변동의 단계적 증폭 현상
    - 작은 흔들림만 있어도 끝부분에서는 커다란 파동이 생기는 현상
- 이처럼 *이해 및 문제 정의가 조금이라도 잘못되면* 최종 인사이트 도출 및 솔루션 적용 단계에서 제대로 된 효과를 보기 힘들다.

> **📌 MECE**
> 

![image](https://github.com/user-attachments/assets/2c5cc26d-827d-458b-a63b-57dc5eb6158d)


- 비즈니스 문제를 올바르게 정의하기 위한 논리적 접근법으로 가장 널리 쓰이는 방법
- 세부 정의들이 서로 겹치지 않고 전체를 합쳤을 때는 빠진 것 없이 완전히 전체를 이루는 것
- 주어진 문제 점을 논리적이고 객관적으로 쪼개어 근본적인 원인을 찾아내는 것
- **로직 트리**를 활용하여 세부 항목 정리
    
    ![image](https://github.com/user-attachments/assets/85d3b2e0-885b-4870-9481-61caf5ae835f)

    
    - 핵심 문제에 대한 세부 문제나 과제 등이 MECE의 원칙에 따라 잘게 뻗어 나가도록 구성
    - 세부 항목들은 서로 중복되지 않으면서 상위 항목 전체를 포함
- 통계적인 모델을 구축하기 위해서는 명확한 분석 모델과 변수가 설정되어야 한다.
- 원활히 진행되기 위해서는 *명확한 문제 정의와 분석 시나리오, 그리고 분석 모델에 적합한 데이터 수집 및 가공 과정이 필요*

> 비즈니스 문제와 목적은 어떻게 해야 잘 정의할 수 있을까?
> 

```
- 통신: 약정 기간이 끝난 고객들이 타 통신사로 이탈하여 회사의 수익이 감소하고 있다
- 유통: 물류창고의 제품을 재고가 부족한 매장으로 운송하는 것이 늦어서 잠재 소비자를 놓치고 있다.
- 금융: 현재의 대출심사 시스템으로 대출을 받은 고객이 대출금을 갚지 못하고 파산하여 
은행이 손해를 보는 경우가 발생하고 있다.
```

- 비즈니스 문제는 현상에 대한 설명으로 끝나서는 안되고, 본질적인 문제점이 함께 전달되어야 하는 것
- 통신사 약정
    - 고객이 이탈하기 때문에 수익이 감소한다는 것이 정확한 비즈니스 문제

![image](https://github.com/user-attachments/assets/95d5ff44-7e9c-4a46-9522-75c56e5db9e4)

- 문제를 **단순한 고객 이탈**로 시작할 경우 → 이탈 고객 예측 모델 만들기
- 문제를 **고객 이탈에 따른 수익 감소**를 문제로 정의한 경우 → 이탈에 따른 손해를 최소화할 수 있는 프로모션 최적화 모델

> **페이오프 매트릭스**
> 

![image](https://github.com/user-attachments/assets/e8c406ce-f1e3-417a-ac5c-52100fa44c87)

- 분석 과제를 도출한 이후에 현재 상황에 맞는 우선순위를 측정하여 프로젝트 과제 수행
- 과제의 수익성과 실행 가능성 수준에 따라 2X2 네 개의 분면에 과제 우선수위를 표현
- Grand Slam
    - 최종 실행과제는 실행가능성과 사업성을 모두 충족
- Extra Innings
- Stolen Base
- Strike Out

## 8.3 분석 목적의 전환

- 분석 프로젝트는 데이터 탐색하기 전까지 데이터에 숨겨져있는 정보와 인사이트를 확인하기 어려우니, 목적을 설정하기 전에 PoC(proof of concept)나 간단한 샘플 데이터 탐색 과정을 거치면 좋음
- 여건이 주어지지 않는 경우가 많기에, *분석 프로젝트 방향이 언제든 바뀔 수 있다는 것을 염두*하기
- 콘셉트가 바뀌는 순간을 인지하고 신속하게 모든 팀원과 실무자들과 공유하기
    - 목적이 불확실해지면 혼란 발생
    - 프로젝트 막바지에 결과를 실무진이나 경영진이 받아들이지 않아 프로젝트가 위기
- 분석 프로젝트를 수행하는 동안에는 실무자들 간의 커뮤니케이션 및 협력이 매우 중요
    - 간단한 상관관계나 데이터 특성 그리고 시각화를 적극 활용
    - 실무자의 관심을 얻고 분석가들이 우리가 하는 데이터를 이해하고, 함께하고 있다는 동질감과 신뢰를 얻기 위함

## 8.4 도메인 지식

- 도메인 지식: 해당되는 분야의 업에 대한 이해도
- 해당 분야의 특성과 프로세스를 제대로 파악하지 못한 상태에서 문제 정의와 분석 목적은 1차원적
- 예시: 배달 플랫폼 도메인
    - 날씨랑 스포츠 경기 시즌에 민감
    - 가게나 메뉴를 정해두고 앱을 켜는 비중이 어느정도 되며, 어떤 행동 패턴을 가지는지에 따라 분석 방향이나 목적이 달라짐
- 직접 의미 있는 변수를 찾아내고 분석 방향을 설정하는 것은 **도메인 지식이 충분하게 있을 때 가능**
- 도메인 지식 습득하기 위한 방법
    - 프로젝트 초반에 잦은 미팅과 함께 적극적인 질문과 자료 요청
    - 관련 논문들을 참고하여 해당 도메인에 대한 심도 있는 지식 습득
    - 현장에 방문해 데이터가 만들어지는 과정을 보기

## 8.5 외부 데이터 수집과 크롤링

- 많은 기업들은 부족한 부분을 보완하고자 외부 데이터를 수집하여 활용
- 수집하기 전에 분석 목적을 명확히 정의하고 수집해야 함
- 실제로 힘들게 외부 데이터 수집 시스템을 구축해 놓고서 정작 제대로 사용하지 않는 경우가 많음
- 절차

![image](https://github.com/user-attachments/assets/4473240e-8986-493d-acc3-e8b15e7abdf2)

- 외부 데이터 수집하는 방법
    
    ![image](https://github.com/user-attachments/assets/6b9bd1dc-bd22-4cd7-b53f-3986f6c5127c)

    - 데이터를 판매하는 기업으로부터 데이터 구매, MOU(양해 각서, 업무 협약석) 등을 통해 데이터 공유
        - 비용이 많이 들고 절차가 복잡
        - 어느 정도 정제된 고품질의 데이터 얻음
    - 공공 오픈 데이터 사이트에서 엑셀이나 csv 형태로 데이터 활용
        - 데이터 수집에 특별한 비용이나 노력이 크게 들어가지 않음
        - 가공하기 위한 리소스가 많이 들어가고 활용성이 높은 데이터를 얻을 확률이 낮음
    - 웹에 있는 데이터를 크롤링하여 수집
        - 원하는 데이터 실시간으로 자유롭게 수집할 수 있다
        - 웹페이지가 리뉴얼되면 이에 맞춰 수집 코드도 수정
        - 기업에서 크롤링을 활용할 때 법적인 이슈도 함께 고려
        - web상을 돌아다니면서 정보를 수집하는 것
        - 스크래핑 - 웹페이지에서 자신이 원하는 부분의 정보만 가져오는 것

> **스크래핑**
> 
- 파이썬에서는 BeautifulSoup이나, Selenium라이브러리
- 웹사이트의 HTML구조를 활용하여 원하는 데이터가 있는 위치를 사전에 설정하여 자동으로 반복적으로 특정 위치에 있는 텍스트를 수집
- 소스 코드를 확인하고, copy → copy selector 선택
- 복사된 위치를 파이썬 코드에 삽입하여 텍스트 수집

# 09. 분석 환경 세팅하기

```
✅ 학습 목표 :
* 데이터 분석의 전체적인 프로세스를 설명할 수 있다.
* 테이블 조인의 개념과 종류를 이해하고, 각 조인 방식의 차이를 구분하여 설명할 수 있다.
* ERD의 개념과 역할을 이해하고, 기본 구성 요소와 관계 유형을 설명할 수 있다.

```

## 9.1 어떤 데이터 분석 언어를 사용하는 것이 좋을까?

> **SAS**
> 
- Statistical Analysis System
- 정확성이 중요한 금융업계 기업들이 많이 사용
- 프로그래밍 스킬이 부족한 사람도 데이터 분석을 하는 것이 어느정도 가능
- 그러나 점점 많이 줄어들고 있음
- 자유도가 높고 비용이 합리적인 R과 파이썬을 사용

> **R**
> 
- 오픈소스지만 시각화 패키지를 통해 효과적으로 시각화 가능
- 활발한 커뮤니티를 활용하여 궁금증 해결
- 프로그래밍적 소양이 부족해서 사용하는데에 큰 무리가 없다

> **파이썬**
> 
- 데이터 분석에 국한되지 않고 다양한 분야에서 활용
- 문법이 쉽고 간단함
- 커뮤니티 활발
- 시각화에서 R에 비해 구현이 복잡하고 직관적이지 못하다
- 데이터를 위해 탄생한 R의 패키지가 매우 강력

![image](https://github.com/user-attachments/assets/4fb25da5-4278-4954-b918-81bd8e5b6000)

> **SQL**
> 
- 대화식 언어이므로 명령문이 짧고 간결
- 데이터 전처리는 SQL과 파이썬 조합
- ML 모델은 사이킷런 등의 패키지 활용

## 9.2 데이터 처리 프로세스 이해하기

> **데이터 흐름**
> 
- OLTP → DW → DM → OLAP

![image](https://github.com/user-attachments/assets/09fddc0c-6c5c-4fb6-9191-8e725931ba67)


- OLTP
    - 실시간으로 데이터를 트랜잭션 단위로 수집, 분류, 저장하는 시스템
    - 데이터가 생성되고 저장되는 처음 단계
- DW(Data Warehouse)
    - 데이터 창고
    - 수집된 데이터를 사용자 관점에서 주제별로 통합하여 원하는 데이터를 쉽게 빼닐 수 있도록 저장
    - OLTP를 보호하고 활용 효율을 높임
    - 전체 히스토리 데이터 보관
    - 비슷한 개념: ODS
        - DW에 저장하기 전에 임시로 데이터를 보관하는 중간 단계의 저장소
        - 최신 데이터를 반영하는 것이 목적
- DM(Data Mart)
    - 사용자의 목적에 맞도록 가공된 일부의 데이터가 저장되는 곳
    - 부서나 사용자 집단의 필요에 맞도록 가공된 개별 데이터 저장소
    - 접근성과 데이터 분석의 효율성을 높이고 시스템 부하 감소
- ETL
    - 추출(Extract): 데이터베이스로부터 필요한 데이터를 읽어 들이는 과정
    - 변환(Transform): 미변환 상태의 raw 데이터를 정리, 필터링, 정형화하고 요약하여 분석에 적합한 상태로 바꾸어 놓는 과정
    - 불러오기(Load): 변환된 데이터를 새로운 테이블(목표 시스템)에 적재하는 과정
    - 저장된 데이터를 사용자가 요구하는 포맷으로 변형하여 이동시키는 과정
- ETL을 통해 연월일 형태로 된 하나의 칼럼으로 변형하여 적재 가능

![image](https://github.com/user-attachments/assets/40d73039-64d5-426f-aec5-9d01a2797ffc)


## 9.3 분산 데이터 처리

- scale-up: 빅데이터를 처리하기 위해 하나의 컴퓨터의 용량을 늘리고 더 빠른 프로세서를 탑재
- scale-out: 분산데이터 처리처럼 여러 대의 컴퓨터를 병렬적으로 연결, 연산 효율이 높다

> **9.3.1 HDFS**
> 
- HDFS - Hadoop Distributed File System
    - 슬레이브 노드: 데이터를 저장하고 계산하는 세부적인 역할
    - 마스터 노드: 대량의 데이터를 HDFS에 저장하고 맵리듀스 방식을 통해 데이터를 병렬 처리
    - 클라이언트 머신: 맵리 듀스 작업을 통해 산출된 결과를 사용자에게 보여주는 역할

**📌 맵리듀스**

- 맵과 리듀스라는 두 단계로 구성
- 맵단계: 흩어져 있는 데이터를 관련된 데이터끼리 묶어서 임시의 집합을 만드는 과정
- 리듀스: 필터링과 정렬을 거쳐 데이터를 뽑아냄
- key-value쌍으로 데이터를 처리

![image](https://github.com/user-attachments/assets/24c85598-5d80-4f6d-9381-e6286f673e35)


- 정렬과 병합 등의 과정을 통해 리듀스 단계에서 나눠져 있던 결과들을 취합하여 최종 결과를 생성
- 이후 다음의 단계를 거쳐 단어의 수를 센다
    - 분할: 입력된 데이터를 고정된 크기의 조각으로 분할
    - 매핑: 분할된 데이터를 key-value 형태로 묶고 단어 개수 계산
    - 셔플링: 매핑 단계의 counting 결과를 정렬 및 병합
    - 리듀싱: 각 결과를 취합 및 계산하여 최종 결괏값을 산출

![image](https://github.com/user-attachments/assets/3c81d43c-0fc5-4520-b35e-938ac6eb23a4)


**📌 하둡**

- 하둡 1.0에서 Job Tracker라는 기본적인 리소스 관리 시스템
    - 전체 클러스터의 리소스 관리
    - 수행 중인 잡들의 진행상황, 에러 관리
    - 완료된 잡들의 로그 저장 및 확인
    
    ![image](https://github.com/user-attachments/assets/5a5ec3f6-00f1-4dc4-83a4-7b8e2d670f9d)

    
- 하둡 2.0은 리소스 매니저와 애플리케이션 마스터 그리고 타임라인 서버 등으로 분리
    - 클러스터마다 애플리케이션 마스터가 존재하여 성공적으로 실행하도록 함
    - 잡에 필요한 자료들은 리소스 매니저를 통해 할당
    - 로그 이력 관리는 타임라인 서버를 통해 진행
    - 노드 매니저는 모든 노드에서 각각의 할당된 태스크를 실행하고 진행 상황 관리

**📌 분산 시스템 구조**

![image](https://github.com/user-attachments/assets/364c815f-3b92-4f6a-abc7-0b1fb9eec907)

- 노드는 하나의 컴퓨터라고 생각
- 몇 개의 컴퓨터가 모인 것이 랙
- 랙들이 모인 것이 클러스터
- 클라이언트가 하나의 잡을 실행하면 여러 개의 태스크를 실행하고 각각의 태스크는 맵과 리듀스를 통해 분산 처리

> **9.3.2 아파치 스파크**
> 

**📌 HDFS와 스파크**

- 스파크는 분산 데이터 처리를 하는 하나의 시스템

![image](https://github.com/user-attachments/assets/62319b77-6bae-4cda-a2b3-4e3b8674cf00)


**📌 스파크의 특징**

- 인메모리 기반의 빠른 데이터 처리가 가능
- Java, Scala, 파이썬, R, SQL 등 다양한 언어를 지원

**📌 제플린**

- 순수 파이썬 언어로 데이터 가공과 모델링 가능
- 스파크의 병렬 처리를 효과적으로 활용하기 위해 다양한 코드 사용 가능

## 9.4 테이블 조인과 정의서 그리고 ERD

- 적어도 3개 이상의 테이블을 조합하고 새로 가공하면서 인사이트 찾기

> **9.4.1 테이블 조인**
> 
- 이너 조인
- 아우터 조인
- 레프트 조인
- 라이트 조인
- 풀 조인
- 크로스 조인

**📌 레프트 조인과 라이트 조인**

![image](https://github.com/user-attachments/assets/c29d8986-1b81-49e9-9de2-8cf81e9a8e87)


- 하나의 테이블을 기준으로 *다른 테이블에서 겹치는 부분*을 결합
- 기준이 되는 테이블의 데이터는 그대로 유지하면서 *조인하는 데이터만 추가*되는 것
- 일치하는 키 값이 없는 행은 조인하는 테이블의 값이 결측값으로 나타남
- 테이블의 키 값에 해당하는 관측치가 여러 개면 그만큼 행이 늘어남

**📌 이너 조인과 풀 조인**

![image](https://github.com/user-attachments/assets/b3430d07-dfc0-480f-9d4b-76983e9056ff)


- 이너 조인: 두 테이블 간에 *겹치는 부분의 행만 가져오는* 조인 방법
    - 부서 코드를 키 값으로 할 때 이너 조인을 한다면 모두 겹치는 행만 나오게 된다
- 풀조인: 반대되는 개념으로 모든 행을 살리는 조인 방법
    - 부서 코드가 없어도 양 테이블의 모든 행이 생성
    - 조인되지 않은 부분은 결측값

**📌 크로스 조인**

- 머신러닝에 사용되는 데이터셋을 생성할 때 사용
- 값이 없더라도 모든 행이 생기도록 데이터 가공을 해야 할 때 크로스 조인 사용

> **9.4.2 데이터 단어 사전**
> 
- 각 칼럼과 테이블의 이름을 정할 때 체계를 약속한 일종의 사전
- 칼럼이나 테이블명을 축약된 단어 형태로 이름을 정함
- 메타 데이터 관리 시스템
    - 데이터가 어디에 어떻게 저장되어 있는지, 데이터를 어떻게 사용할 것인지 이해할 수 있도록 데이터에 대한 정보를 관리하는 시스템

> **9.4.3 테이블 정의서**
> 
- 각 DW, DM 등에 적재된 테이블과 칼럼의 한글과 영문명, 데이터 속성, 그리고 간단한 설명이 정리된 표
- 엑셀 파일로 만들어서 곧바로 원하는 정보를 찾아보기 위해 사용
- 처음 데이터 환경을 이해하기 위해서 ERD를 먼저 봐야함

> **9.4.4 ERD**
> 
- Entity Relationship Diagram로, 테이블의 구성 정보와 테이블 간 관계를 도식으로 표현한 그림 형태

![image](https://github.com/user-attachments/assets/a82198df-9ccb-4e40-9513-0691292343b9)

- 물리: 영문
    - DB를 효율적이고 결점없이 구현하는 것을 목표로 구현하는 개념
- 논리: 한글
    - 사용자 입장에서 테이블 간 매핑에 오류가 없으면 정규화가 이뤄진 ERD의 개념
- 논리 ERD를 보고 DB 구조 파악
- 핵심: 테이블 간 연결을 해주는 **키 칼럼**과 연결 관계를 의미하는 **식별자**
- 키 칼럼
    - 기본 키: 테이블에 적재된 각각의 데이터를 유일하게 구분하는 키
        - 결측 값을 가질 수 없다
    - 외래 키: 각 테이블 간에 연결을 만들기 위해서 테이블에서 다른 테이블의 참조되는 기본 키
        - 중복이나 결측 값 존재 가능
        - 외래 키가 정의된 테이블은 자식 테이블, 참조되는 테이블은 부모 테이블

**📌 테이블 간 연결 관계를 나타내는 규칙**

![image](https://github.com/user-attachments/assets/42d67c01-2d67-4d48-93b0-af2d15ea5f61)

- 테이블 간에는 1:1로 매칭되는 경우가 있고 1:N, N:N등으로 연결된 경우도 많기 때문에 이러한 관계를 정확히 파악하고 데이터를 다뤄야 함

# 확인 문제

## 문제 1.

> 🧚 아래의 테이블을 조인한 결과를 출력하였습니다. 어떤 조인 방식을 사용했는지 맞춰보세요.
> 

> 사용한 테이블은 다음과 같습니다.
> 

[제목 없음](https://www.notion.so/1cae3b9c30be801cb8d8edb0a2afe589?pvs=21)

> 보기: INNER, LEFT, RIGHT 조인
> 

<!-- 테이블 조인의 종류를 이해하였는지 확인하기 위한 문제입니다. 각 테이블이 어떤 조인 방식을 이용하였을지 고민해보고 각 테이블 아래에 답을 작성해주세요.-->

### 1-1.

![](https://github.com/ejejbb/Template/raw/main/File/2-1.PNG)

```
1007 행의 dep_cd와 dep_nm, location이 결측값이라 emp_cd와 emp_nm, job은 그대로 있고, 뒤 3개의 열이 추가된 left join이라고 생각한다. 
```

### 1-2.

![](https://github.com/ejejbb/Template/raw/main/File/2-3.PNG)

```
모든 칼럼에 값이 다 채워져있기 때문에 겹치는 값만 넣은 inner join이라고 생각한다.
```

### 1-3.

![](https://github.com/ejejbb/Template/raw/main/File/2-2.PNG)

```
emp_cd, emp_nm, job이 결측값이기 때문에 dep_cd, dep_nm, location은 그대로 있고 앞 3개 열이 추가된 right join이라고 생각한다.
```

### 🎉 수고하셨습니다.
