#1. 데이터 가져오기
from sklearn.datasets import load_diabetes

diabetes = load_diabetes()
df_X = diabetes.data
df_y = diabetes.target


#2.& 3. 모델에 입력할 데이터 X y 준비하기 (이미 <class 'numpy.ndarray'> 이므로 따로 변환 X)
print("X's type : "+str(type(df_X)))
print("y's type : "+str(type(df_y)))

#4. train, test 분리하기
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.2, random_state=2025)

#5. 모델 준비하기
import numpy as np

print("X.shape : " + str(df_X.shape)) # 10개의 피처 확인
print("y.shape : " + str(df_y.shape))
num_features = df_X.shape[1]
W = np.random.randn(num_features)
b = np.random.randn()


def Model(X, W=W, b=b):
    predict = np.dot(X, W) + b
    return predict

#6. 손실함수 loss 정의하기

def Loss(label, predict):
    n = label.shape[0]
    return np.sum((label - predict)**2) / n

#7. 기울기를 구하는 gradient 함수 구현하기

def gradient(X, label, predict):
    n = label.shape[0]
    dW = 2 * np.dot(X.T, (predict - label)) / n
    db = 2 * np.sum(predict - label) / n
    return dW, db

#8. 하이퍼 파라미터인 학습률 정하기
learning_rate = [0.01, 0.1, 1]

#9. 모델 학습하기
epoch = 1000
losses_1, losses_2, losses_3 = [], [], []

for lr in learning_rate:
    W_copy = W  # 새로 초기화
    b_copy = b
    losses = []
    for _ in range(epoch):
        prediction = Model(X_train, W=W_copy, b=b_copy)
        loss = Loss(y_train, prediction)
        losses.append(loss)
        dW, db = gradient(X_train, y_train, prediction)
        W_copy -= lr * dW
        b_copy -= lr * db

    # 손실값 기록
    if lr == 0.01:
        losses_1 = losses
        W_1, b_1 = W_copy, b_copy
    elif lr == 0.1:
        losses_2 = losses
        W_2, b_2 = W_copy, b_copy
    else:
        losses_3 = losses
        W_3, b_3 = W_copy, b_copy

#10. test데이터에 대한 성능 확인하기
test_prediction_1 = Model(X_test, W=W_1, b=b_1)
test_prediction_2 = Model(X_test, W=W_2, b=b_2)
test_prediction_3 = Model(X_test, W=W_3, b=b_3)

loss_1 = Loss(y_test,test_prediction_1)
loss_2 = Loss(y_test,test_prediction_2)
loss_3 = Loss(y_test,test_prediction_3)

print("==========================================================================")
print(f"Loss in test with learning rate = 0.01 : {loss_1} \n  W : {W_1}\n  b : {b_1}\n")
print(f"Loss in test with learning rate = 0.1 : {loss_2} \n  W : {W_2}\n  b : {b_2}\n")
print(f"Loss in test with learning rate = 1 : {loss_3} \n  W : {W_3}\n  b : {b_3}\n")

#시각화
import matplotlib.pyplot as plt
plt.plot(losses_1, label='lr=0.01')
plt.plot(losses_2, label='lr=0.1')
plt.plot(losses_3, label='lr=1')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Comparing different Learning rate')
plt.legend()
plt.show()
