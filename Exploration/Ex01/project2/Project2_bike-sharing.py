#1. 데이터 가져오기기
import pandas as pd
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#2. datetime 칼럼을 datetime 자료형으로 변환하고 연,월,일,시,분,초까지 6가지 칼럼 생성하기

# datetime 컬럼을 datetime 타입으로 변환
train['datetime'] = pd.to_datetime(train['datetime'])

# 새로운 시간 관련 컬럼 생성
train['year'] = train['datetime'].dt.year
train['month'] = train['datetime'].dt.month
train['day'] = train['datetime'].dt.day
train['hour'] = train['datetime'].dt.hour
train['minute'] = train['datetime'].dt.minute
train['second'] = train['datetime'].dt.second

#3. year, month, day, hour, minute, second 데이터 개수 시각화하기
import matplotlib.pyplot as plt
import seaborn as sns

new_col = ['year','month','day','hour','minute','second']

fig, axes = plt.subplots(3, 2, figsize=(10,10))
for i, col in enumerate(new_col):
    ax = axes[i//2, i%2]
    sns.countplot(data=train, x=col, ax=ax)  # 각 서브플롯에 countplot
    ax.set_title(f'Countplot of {col}')

plt.tight_layout()
plt.show()

#4. X,y 칼럼 선택 및 train/test 데이터 분리

#앞서 새로운 칼럼을 시각화 했을 때, minute과 second는 모두 동일하여 의미가 없음을 확인인

train['year_month'] = ((train['year']-2011) * 12) + train['month']

# 연도별 평균 대여량
yearly_count = train.groupby('year_month')['count'].mean()

plt.figure(figsize=(8, 4))
plt.plot(yearly_count.index, yearly_count.values, marker='o')
plt.title('year_month Average Bike Rentals')
plt.xlabel('Month')
plt.ylabel('Average Count')
plt.xticks(range(1, 25))
plt.grid(True)
plt.show()
#년도에 따라, 바이크를 대여하는 수가 늘어나고 있음으로 렌탈 수치 예측에 년도에 따른 인플레이션을 반영하는 방법도 고려해볼만함함
# 4~10월이 성수기 / 11~3월 비수기인 것을 확인 할 수 있음

# 시간대별 평균 대여량
hourly_count = train.groupby('hour')['count'].mean()
plt.figure(figsize=(8, 4))
plt.plot(hourly_count.index, hourly_count.values, marker='o', color='green')
plt.title('Hourly Average Bike Rentals')
plt.xlabel('Hour')
plt.ylabel('Average Count')
plt.grid(True)
plt.xticks(range(0, 24))
plt.show()
# 시간대에 따라 대여하는 정도가 차이나는 것을 확인 7시~20시 / 21시~6시 혹은 더 세분하게 범주형 변수로 사용하는 것이 유의미할 것으로 예상

#요일별 평균 대여량 비교
train['weekday'] = train['datetime'].dt.weekday  # 0: 월요일, 6: 일요일
train['is_weekend'] = train['weekday'].apply(lambda x: 1 if x >= 5 else 0)  # 주말 여부 (0: 평일, 1: 주말)
weekday_count = train.groupby('weekday')['count'].mean()
plt.figure(figsize=(8, 4))
plt.bar(weekday_count.index, weekday_count.values, color='purple')
plt.title('Weekday Average Bike Rentals')
plt.xlabel('Weekday (0: Monday, 6: Sunday)')
plt.ylabel('Average Count')
plt.grid(axis='y')
plt.xticks(range(0, 7), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
plt.show()
# 요일별로 관찰했을때 약간의 차이가 존재하긴 하지만 예측에 유의미할 지 알 수 없음

# 주중/주말 평균 대여량
weekend_count = train.groupby('is_weekend')['count'].mean()
plt.figure(figsize=(6, 4))
plt.bar(['Weekday', 'Weekend'], weekend_count.values, color=['blue', 'orange'])
plt.title('Average Bike Rentals: Weekday vs Weekend')
plt.ylabel('Average Count')
plt.grid(axis='y')
plt.show()

#월~금 과 토~일의 차이 역시 근소함


# workingday와 holiday가 반대 관계에 있는지 확인
relationship = train.groupby(['workingday', 'holiday']).size().reset_index(name='count')

# 관계 확인-> 서로 반대 관계가 아닌 것을 확인인
print(train.shape)
print(relationship)
# workingday와 holiday가 모두 0인 데이터 필터링
weekend_data = train[(train['workingday'] == 0) & (train['holiday'] == 0)]

# weekend_data의 datetime 열 확인
weekend_data[['datetime']].head()

# datetime 열을 datetime 형식으로 변환
weekend_data['datetime'] = pd.to_datetime(weekend_data['datetime'])

# 요일(0: 월요일, 1: 화요일, ..., 6: 일요일) 추출
weekend_data['weekday'] = weekend_data['datetime'].dt.weekday

# 주말(토요일, 일요일)인 데이터 확인
print(weekend_data[['datetime', 'weekday']].describe())

#두 피처의 수가 똑같음 : workingday와 hoilday가 모두 0인 경우는 토요일 혹은 일요일임을 확인

# workingday가 1인 경우와 0인 경우에 대한 평균 대여량 계산
workingday_comparison = train.groupby('workingday')['count'].mean().reset_index()

# 시각화
plt.figure(figsize=(6, 4))
sns.barplot(x='workingday', y='count', data=workingday_comparison)
plt.title('Average Rentals by Workingday')
plt.xlabel('(0: Free-Workday, 1: Workingday)')
plt.ylabel('Average Count')
plt.xticks([0, 1], ['Free-Workday', 'Workingday'])
plt.show()
#근소한 차이긴 하지만 주중/주말로 비교했을 때보다, 일하는 날/노는 날 로 비교했을 때 렌탈하는 경우가 더 차이가 나는 것을 확인



#시계열 패턴을 통한 EDA 정리
# 1. count 예측 시, 년도 별 count의 인플레이션을 고려해야할 필요가 있음, 모델링 시 인플레이션을 반영하거나 'Year'피처를 사용
# 2. 요일별 혹은 주말, 주중, 쉬는날 노는날로 비교했을 때 큰 차이를 확인하기 어려웠음음
# 3. 월별, 시간대별을 관찰했을 때 구간별 차이가 유의미하게 존재하므로 'month'와 'hour' 학습에 피처 활용


#범주형 변수 분석
# 범주형 변수별 빈도수 계산
#season은 결국 month를 4단위로 쪼갠 내용과 동일 ; count와의 관계를 고려하여 month를 적절히 쪼갤 수 있다면 season을 사용하는 것보다 낫다고 생각
#범주형 변수 중 결국 사용할 변수는 weather라고 생각
categorical_vars = ['season', 'weather']

for var in categorical_vars:
    print(f"{var} value counts:")
    print(train[var].value_counts(), "\n")



# 범주형 변수별 데이터 분포
plt.figure()
for i, var in enumerate(categorical_vars):
    plt.subplot(1, 2, i+1)
    sns.countplot(x=var, data=train, palette="pastel")
    plt.title(f"Distribution of {var}")
    plt.xlabel(var)
    plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
#weather 역시 1에 대부분으 데이터가 몰려있고 2,3의 비중이 적고 4는 없다고 봐도 무관
"""
1 - 구름이 없거나 약간의 구름
2 - 약한 안개와 구름
3 - 약한 눈 혹은 강우, 때때로 천둥을 동반
4 - 심한 비, 눈, 안개
"""
#weather의 정의를 보면 4에 자전거 대여량이 늘 수 없는 상황
plt.figure()

for i, var in enumerate(categorical_vars):
    plt.subplot(1, 2, i+1)
    sns.barplot(x=var, y='count', data=train, palette="viridis")
    plt.title(f"Average Rentals by {var}")
    plt.xlabel(var)
    plt.ylabel("Average Count")

plt.tight_layout()
plt.show()
#weather에 따른 count를 살펴보면 1,2,3으로 갈수록 적어지고 4에서 갑자기 커지지만 이는 outlier로 판단되며 적절한 변환이 필요함 (무엇으로 대체하는게 적절할까.. 우선 4를 3으로 변경하여 예측을 진행)



#범주형 변수 EDA 정리
#1. weather가 결국 핵심 피처로 사용하며 4는 outlier로 4->3으로 변경하여 처리



#연속형 변수 분석
continuous_vars = ['temp', 'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count']

# 히스토그램과 KDE 그리기
plt.figure(figsize=(10, 10))
for i, var in enumerate(continuous_vars):
    plt.subplot(3, 3, i+1)
    sns.histplot(train[var], kde=True, color="skyblue", bins=30)
    plt.title(f'Distribution of {var}')
plt.tight_layout()
plt.show()
#온도,습도,풍속,체감온도는 서로 어느정도의 상관관계가 있을 것으로 추측하고 이를 조합하거나 제거하는 방식을 사용해서 다중공선성을 줄여야 할 것으로 예상
#casual과 registered의 합이 count와 일치할 것으로 예상되는데 casual과 registered 상황을 각각 예측해서 최종 count로 합치는게 더 나은 성능을 보여줄지 잘 모르겠음, 우선 둘을 합친게 count이니 두 피처는 무시하는 걸로 진행

# 연속형 변수만 추출하여 상관계수 계산
correlation_matrix = train[continuous_vars].corr()

# Heatmap 시각화(상관관계 확인)
plt.figure(figsize=(10, 7))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Continuous Variables')
plt.show()

# Scatterplot 시각화(분포, 상관관계 확인)
plt.figure(figsize=(15, 10))
for i, var in enumerate(continuous_vars[:-1]):  # 마지막 변수인 'count'는 제외
    plt.subplot(3, 3, i+1)
    sns.scatterplot(x=train[var], y=train['count'], alpha=0.6, color='teal')
    plt.title(f'{var} & Count')
plt.tight_layout()
plt.show()

#연속형 변수 EDA 정리
#1. windspeed의 경우, 0이거나 7이거나로 나뉘는 것으로 보아 범주화가 필요해 보임 (사분위수로 나누는게 적절하지 않나?)
#2. 온도,체감온도,습도,풍속의 경우 앞선 변수 중 weather, season과 상관관계가 높아 선택적으로 활용하거나 새롭게 조합해서 활용하는게 좋지 않을까 싶음
#3. registered와 casual의 경우 둘의 합이 count이기 때문에 두 피처는 제거


#EDA 총정리
#시계열 패턴을 통한 EDA 정리
# 1. count 예측 시, 년도 별 count의 인플레이션이 눈에 띄므로 "Year" 피처가 중요함 -> 2011 : 0 , 2012 : 1 로 매핑하여 처리하면 될 듯
# 2. 요일별, 주말, 주중, 쉬는날, 노는날은 다른 시계열 피처들보다 눈에 띄는 차이가 없음 : 사용 X
# 3. 월별, 시간대별을 관찰했을 때 구간별 차이가 존재함-> month는 4~10 : 1 (성수기), 11~3 : 0(비수기) hour은 7~20 : 1 (주 대여 시간) 21~6 : 0 (반납 시간) 으로 매핑
#범주형 변수 EDA 정리
#1. weather가 결국 핵심 피처, outlier는 4->3으로 변경하여 처리 , 
#2. 앞서 밝혔듯이 workingday와 hoilday의 분포차이가 심함, 앞서서는 upsampling,downsampling을 고려하였지만
#woringday와 hoilday의 count 차이가 심하지 않고 오히려 workingday와 workingday가 아닌날의 차이가 심함으로 workingday만 피처로 사용하면 될 것으로 정리
#연속형 변수 EDA 정리
#1. 예측하고자 하는 count가 정규분포를 따르지 않기 때문에 비선형 회귀 모델을 선정할 것
#2. windspeed의 경우, 0이거나 7이거나로 나뉘는 것으로 보아 범주화가 필요해 보임 (quantile값에 따라 4 단위정도로 나누는게 적절하지 않나?)
#3. 온도,체감온도,습도,풍속의 경우 앞선 변수 중 weather, season과 상관관계가 높아 선택적으로 활용하거나 새롭게 조합해서 활용하는게 좋지 않을까 싶음
#4. registered와 casual의 경우 둘의 합이 count이기 때문에 바로 count를 예측하는게 나을지 둘을 따로 예측하고 이 예측을 다시 count의 예측으로 활용하는게 좋을지 애매함


# windspeed_group 생성 및 windspeed 제거
def categorize_windspeed(ws):
    if ws == 0:
        return 2
    elif ws <= 30:
        return 1
    else:
        return 0

train = pd.read_csv('/kaggle/input/bike-sharing-demand/train.csv')
test = pd.read_csv('/kaggle/input/bike-sharing-demand/test.csv')

train['windspeed_group'] = train['windspeed'].apply(categorize_windspeed)
test['windspeed_group'] = test['windspeed'].apply(categorize_windspeed)
test['windspeed_group'].unique()

# weather == 4라면 3으로 변경 , 0,1 로 그룹핑

train['weather'] = train['weather'].replace(2, 1)
train['weather'] = train['weather'].replace(3, 0)
train['weather'] = train['weather'].replace(4, 0)

test['weather'] = test['weather'].replace(2, 1)
test['weather'] = test['weather'].replace(3, 0)
test['weather'] = test['weather'].replace(4, 0)

# month_group 생성 (기존 season은 계절중심, month_group은 count에 따라 성수기 비수기)
def categorize_month(ws):
    if 4 <= ws <= 10:
        return 1
    else:
        return 0

train['month_group'] = train['month'].apply(categorize_month)
test['month_group'] = test['month'].apply(categorize_month)

# hour_group 생성 (month_group과 유사하게 hour_group)
def categorize_hour(ws):
    if 7 <= ws <= 20:
        return 1
    else:
        return 0

train['hour_group'] = train['hour'].apply(categorize_hour)
test['hour_group'] = test['hour'].apply(categorize_hour)


feature = ['year','month_group', 'hour_group', 'weather', 'humidity', 'windspeed_group']


#(5) LinearRegression 모델 학습
#sklearn의 LinearRegression 모델 불러오기 및 학습하기
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)


#(6) 학습된 모델로 X_test에 대한 예측값 출력 및 손실함수값 계산
#학습된 모델에 X_test를 입력해서 예측값 출력하기
#모델이 예측한 값과 정답 target 간의 손실함수 값 계산하기
#mse 값과 함께 rmse 값도 계산하기

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"MSE (Mean Squared Error): {mse:.2f}")
print(f"RMSE (Root Mean Squared Error): {rmse:.2f}")

#(7) x축은 temp 또는 humidity로, y축은 count로 예측 결과 시각화하기
#x축에 X 데이터 중 temp 데이터를, y축에는 count 데이터를 넣어서 시각화하기
#x축에 X 데이터 중 humidity 데이터를, y축에는 count 데이터를 넣어서 시각화하기

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(X_test['temp'], y_test, color='blue', label='Actual')
plt.scatter(X_test['temp'], y_pred, color='red', label='Predicted', alpha=0.7)
plt.title('Temperature & Count')
plt.xlabel('Temperature')
plt.ylabel('Count')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(X_test['humidity'], y_test, color='blue', label='Actual')
plt.scatter(X_test['humidity'], y_pred, color='red', label='Predicted', alpha=0.7)
plt.title('Humidity & Count')
plt.xlabel('Humidity')
plt.ylabel('Count')
plt.legend()
