import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.style.use("fivethirtyeight")
%matplotlib inline
df=pd.read_csv('/kaggle/input/iris-csv/Iris (1).csv')
df.head()
df.info()   
df.describe()
df.shape
df.drop('Id',axis=1,inplace=True)
df.head()
df['Species'].value_counts()

df.isnull().sum()
import missingno as msno
msno.bar(df)
df.drop_duplicates(inplace=True)
sns.pairplot(df, hue="Species", size=3)
import pandas.plotting
from pandas.plotting import andrews_curves
andrews_curves(df, "Species")

X=df.drop('Species',axis=1)
y=df['Species']
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
df['Species'] = pd.Categorical(df.Species)
df['Species'] = df.Species.cat.codes

y = to_categorical(df.Species)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,stratify=y,random_state=123)
model=Sequential()
model.add(Dense(100,activation='relu',input_shape=(4,)))

model.add(Dense(3,activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
history=model.fit(X_train,y_train,epochs=45,validation_data=(X_test, y_test))
model.evaluate(X_test,y_test)
pred = model.predict(X_test[:10])
print(pred)
p=np.argmax(pred,axis=1)
print(p)
print(y_test[:10])
history.history['accuracy']
history.history['val_accuracy']
plt.figure()


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'])
plt.show()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 150 entries, 0 to 149
Data columns (total 6 columns):
 #   Column         Non-Null Count  Dtype  
---  ------         --------------  -----  
 0   Id             150 non-null    int64  
 1   SepalLengthCm  150 non-null    float64
 2   SepalWidthCm   150 non-null    float64
 3   PetalLengthCm  150 non-null    float64
 4   PetalWidthCm   150 non-null    float64
 5   Species        150 non-null    object 
dtypes: float64(4), int64(1), object(1)
memory usage: 7.2+ KB
Epoch 1/45
4/4 ━━━━━━━━━━━━━━━━━━━━ 1s 47ms/step - accuracy: 0.3083 - loss: 1.3766 - val_accuracy: 0.3333 - val_loss: 1.2008
Epoch 2/45
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - accuracy: 0.3458 - loss: 1.1429 - val_accuracy: 0.3333 - val_loss: 1.0882
Epoch 3/45
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - accuracy: 0.3833 - loss: 1.0792 - val_accuracy: 0.3667 - val_loss: 1.0347
Epoch 4/45
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.3686 - loss: 1.0224 - val_accuracy: 0.3333 - val_loss: 0.9900
Epoch 5/45
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.3233 - loss: 0.9961 - val_accuracy: 0.5667 - val_loss: 0.9352
Epoch 6/45
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.6409 - loss: 0.9160 - val_accuracy: 0.4667 - val_loss: 0.8906
Epoch 7/45
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.5033 - loss: 0.8848 - val_accuracy: 0.6667 - val_loss: 0.8545
Epoch 8/45
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.7080 - loss: 0.8357 - val_accuracy: 0.6667 - val_loss: 0.8219
Epoch 9/45
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - accuracy: 0.6695 - loss: 0.8043 - val_accuracy: 0.6667 - val_loss: 0.7868
Epoch 10/45
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - accuracy: 0.7146 - loss: 0.7660 - val_accuracy: 0.7000 - val_loss: 0.7552
Epoch 11/45
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - accuracy: 0.7667 - loss: 0.7663 - val_accuracy: 0.9000 - val_loss: 0.7294
Epoch 12/45
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - accuracy: 0.9859 - loss: 0.7461 - val_accuracy: 0.9000 - val_loss: 0.7055
Epoch 13/45
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.9835 - loss: 0.7139 - val_accuracy: 0.8000 - val_loss: 0.6821
Epoch 14/45
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - accuracy: 0.8597 - loss: 0.6840 - val_accuracy: 0.7000 - val_loss: 0.6625
Epoch 15/45
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.7536 - loss: 0.6607 - val_accuracy: 0.7000 - val_loss: 0.6433
Epoch 16/45
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - accuracy: 0.7247 - loss: 0.6649 - val_accuracy: 0.8000 - val_loss: 0.6204
Epoch 17/45
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.8508 - loss: 0.6324 - val_accuracy: 0.8333 - val_loss: 0.6018
Epoch 18/45
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.9149 - loss: 0.6011 - val_accuracy: 0.8333 - val_loss: 0.5863
Epoch 19/45
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.9061 - loss: 0.5948 - val_accuracy: 0.8333 - val_loss: 0.5716
Epoch 20/45
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.9186 - loss: 0.5797 - val_accuracy: 0.8333 - val_loss: 0.5582
Epoch 21/45
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.9073 - loss: 0.5564 - val_accuracy: 0.8333 - val_loss: 0.5459
Epoch 22/45
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.9054 - loss: 0.5727 - val_accuracy: 0.8667 - val_loss: 0.5335
Epoch 23/45
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.9519 - loss: 0.5475 - val_accuracy: 0.8667 - val_loss: 0.5220
Epoch 24/45
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.9456 - loss: 0.5385 - val_accuracy: 0.8333 - val_loss: 0.5124
Epoch 25/45
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - accuracy: 0.9228 - loss: 0.5159 - val_accuracy: 0.8333 - val_loss: 0.5019
Epoch 26/45
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.9127 - loss: 0.5026 - val_accuracy: 0.8667 - val_loss: 0.4918
Epoch 27/45
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.9268 - loss: 0.4835 - val_accuracy: 0.8667 - val_loss: 0.4830
Epoch 28/45
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.9354 - loss: 0.4939 - val_accuracy: 0.9000 - val_loss: 0.4736
Epoch 29/45
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.9655 - loss: 0.4659 - val_accuracy: 0.9000 - val_loss: 0.4647
Epoch 30/45
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.9624 - loss: 0.4712 - val_accuracy: 0.9000 - val_loss: 0.4563
Epoch 31/45
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.9592 - loss: 0.4604 - val_accuracy: 0.9000 - val_loss: 0.4480
Epoch 32/45
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - accuracy: 0.9610 - loss: 0.4429 - val_accuracy: 0.8667 - val_loss: 0.4420
Epoch 33/45
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.9319 - loss: 0.4630 - val_accuracy: 0.9000 - val_loss: 0.4333
Epoch 34/45
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - accuracy: 0.9624 - loss: 0.4255 - val_accuracy: 0.9000 - val_loss: 0.4260
Epoch 35/45
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.9519 - loss: 0.4317 - val_accuracy: 0.9333 - val_loss: 0.4194
Epoch 36/45
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.9710 - loss: 0.4235 - val_accuracy: 0.9667 - val_loss: 0.4128
Epoch 37/45
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.9731 - loss: 0.4266 - val_accuracy: 0.9333 - val_loss: 0.4065
Epoch 38/45
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.9710 - loss: 0.4206 - val_accuracy: 0.9333 - val_loss: 0.4004
Epoch 39/45
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.9728 - loss: 0.3817 - val_accuracy: 0.9000 - val_loss: 0.3969
Epoch 40/45
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.9613 - loss: 0.4033 - val_accuracy: 0.9000 - val_loss: 0.3891
Epoch 41/45
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.9741 - loss: 0.3741 - val_accuracy: 0.9667 - val_loss: 0.3830
Epoch 42/45
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.9796 - loss: 0.3913 - val_accuracy: 0.9667 - val_loss: 0.3777
Epoch 43/45
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.9616 - loss: 0.3731 - val_accuracy: 0.9333 - val_loss: 0.3729
Epoch 44/45
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - accuracy: 0.9676 - loss: 0.3622 - val_accuracy: 0.9000 - val_loss: 0.3699
Epoch 45/45
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - accuracy: 0.9679 - loss: 0.3638 - val_accuracy: 0.9333 - val_loss: 0.3629
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step - accuracy: 0.9333 - loss: 0.3629
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 35ms/step
[[0.08084621 0.610276   0.3088778 ]
 [0.01211291 0.3419217  0.64596534]
 [0.87606764 0.11408616 0.00984623]
 [0.93472517 0.06162326 0.00365174]
 [0.0250315  0.4135101  0.5614583 ]
 [0.00604901 0.29320773 0.70074326]
 [0.86422503 0.12375108 0.01202394]
 [0.02134737 0.45922014 0.5194325 ]
 [0.9242806  0.07024986 0.00546963]
 [0.0103968  0.34707043 0.6425328 ]]
[1 2 0 0 2 2 0 2 0 2]
[[0. 1. 0.]
 [0. 0. 1.]
 [1. 0. 0.]
 [1. 0. 0.]
 [0. 0. 1.]
 [0. 0. 1.]
 [1. 0. 0.]
 [0. 1. 0.]
 [1. 0. 0.]
 [0. 0. 1.]]


