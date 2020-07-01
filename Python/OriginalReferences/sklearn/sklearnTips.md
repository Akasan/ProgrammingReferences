# sklearn

- preprocessingはデータの前処理に用いるクラス

- import
```python
from sklearn import preprocessing
```

## Binarization 
Binarizerによってデータをバイナリに変換する

```python
data=np.array(...)
output=preprocessing.Binarizer(threshold=num).transform(data)
```
ここで、numは閾値を設定する。変換は基本列単位

## Mean removal
Mean removalは、平均0、標準偏差が1になるように変換する。変換は基本列単位
```python
data_scaled=preprocessing.scale(input_data)
data_scaled.mean() #平均
data_scaled.std() #標準偏差
```

## Scaling
axisを列単位で見た場合に、最大値が1,最小値が０になるように線形変換

```python
data_scaler_minmax=preprocessing.MinMaxScaler(feature_range=(min,max))
data_scaled_minmax=data_scaler_minmax.fit_transform(input_data)
```
入力データを[min,max]の範囲で線形変換

## Normalization
L1ノルムやL2ノルムにて列方向で正規化

```python
data_normalized_l1 = preprocessing.normalize(input_data, norm='l1')
data_normalized_l2 = preprocessing.normalize(input_data, norm='l2')
```

## Label Encoding
例えば、次のようなデータ群があったとする。
```python
input_data=["red","green","green","yellow","black","blue","red"]
```
これらのデータを重複している要素をふまえてエンコーディングするには次のように行う。

```python
encoder=preprocessing.LabelEncoder()
encoder.fit(input_data)
for i ,item in enumerate(encoder.classes_):
    print(i,item)
    
test_label=[...]
encoded_output=encoder.transform(test_label)   # Encoding

encoded_label=[...]
decoded_output=encoder.inverse_transform(encoded_label)  # Decoding
```
これにより、encoderオブジェクトはinput_dataにあるデータをエンコードできるようになる。基本的には昇順にデータを表示ならべてエンコードされる。なので、今回の例で言うとアルファベット順に考えてblackを0として順に数値が当てはめられる。エンコードする場合はencoder.transform()を、デコードする場合はencoder.inverse_transform()を利用する。

## Logistic Regression
```python
from sklearn import linear_model
classifier=linear_model.LogisticRegression(solver="liblinear",C=1)
classifier.fit(X,y)
```

## Naive Bayes 
```python
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation  #This will be removed

X_train,X_test,y_train,y_test=cross_validation.train_test_split(X,y,test_size=0.2,random_state=3)

classifier=GaussianNB()
classifier.fit(X_train,y_train)  # Model fitting with training data

y_pred=classifier.predict(X_test)
accuracy=100.0*(y_pred==y_test).sum()/X.shape[0]

# calculate accuracy and so on
accuracy_values = cross_validation.cross_val_score(classifier,
           X, y, scoring='accuracy', cv=num_folds)
print("Accuracy: " + str(round(100*accuracy_values.mean(), 2)) + "%")

precision_values = cross_validation.cross_val_score(classifier,
           X, y, scoring='precision_weighted', cv=num_folds)
print("Precision: " + str(round(100*precision_values.mean(), 2)) + "%")

recall_values = cross_validation.cross_val_score(classifier,
           X, y, scoring='recall_weighted', cv=num_folds)
print("Recall: " + str(round(100*recall_values.mean(), 2)) + "%")
 
f1_values = cross_validation.cross_val_score(classifier,
           X, y, scoring='f1_weighted', cv=num_folds)
print("F1: " + str(round(100*f1_values.mean(), 2)) + "%")
```

## Confusion matrix
- True positive 
the samples for which we predicted 1 as the output and the gruond truth is 1 .
- True negative
the samples for which we predicted 0 as the output and the gruond truth is 0 .
- False positive
the samples for which we predicted 1 as the output and the gruond truth is 0 .
- False negative
the samples for which we predicted 0 as the output and the gruond truth is 1 .

# sklearnを利用した機械学習
- 線形回帰分析の一例
```python
# 必要なモジュールのインポート
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# データの生成
X, y = make_regression(n_samples=100, n_features=1, n_targets=1, noise=5.0, random_state=42)
train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=42)

# 以下にコードを記述してください。
model=LinearRegression()

model.fit(train_X,train_y)

# test_X, test_yに対する決定係数を出力してください
print(model.score(train_X,train_y))
```

- make_regression()
・n_samples:データ数
・n_features:変数の数
