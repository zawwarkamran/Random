import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.metrics import confusion_matrix
import statsmodels.api as sm
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
np.seterr('ignore')

de = pd.read_csv('default.csv')
df = pd.read_csv('default.csv')
df.drop('Unnamed: 0', axis=1, inplace=True)
de.drop(['Unnamed: 0', 'student', 'income'], axis=1, inplace=True)
de['default'].replace(to_replace=['No', 'Yes'], value=[0, 1], inplace=True)
df['default'].replace(to_replace=['No', 'Yes'], value=[0, 1], inplace=True)
df['student'].replace(to_replace=['No', 'Yes'], value=[0, 1], inplace=True)
observ = df.drop('default', axis=1).values
respon = df['default'].values.reshape(-1,1)

# LogisticReg = sm.GLM(endog=de['default'], exog=sm.add_constant(de['balance']), family=sm.families.Binomial()).fit()


def log_likelihood(par):
    xb = par[0] + par[1]*de['balance']
    return -np.sum((de['default']*xb)-np.log(1+np.exp(xb)))


MLE = minimize(log_likelihood, x0=np.array([0, 0]), method='Nelder-Mead')
print(MLE)

# Bernoulli
values = np.array([1, 1, 1, 0, 0])


def bernloglik(par):
    return -np.sum(values*np.log(par)+(1-values)*np.log(1-par))


MLE_2 = minimize(bernloglik, x0=np.array([0]), method='Nelder-Mead')

parvals = np.linspace(0, 1, 20)
funcvals = list(map(lambda x: -bernloglik(x), parvals))

# plt.plot(parvals, funcvals)
# plt.show()

sc = StandardScaler()
ohe = OneHotEncoder()
observ = sc.fit_transform(observ)
respon = ohe.fit_transform(respon).toarray()
print(respon)

X_train, X_test, y_train, y_test = train_test_split(observ, respon, test_size=0.1)

model = Sequential()
model.add(Dense(3, input_dim=3, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=100, batch_size=64)

y_pred = model.predict(X_test)
#Converting predictions to label
pred = list()
for i in range(len(y_pred)):
    pred.append(np.argmax(y_pred[i]))
#Converting one hot encoded test label to label
test = list()
for i in range(len(y_test)):
    test.append(np.argmax(y_test[i]))

from sklearn.metrics import accuracy_score
a = accuracy_score(pred,test)
print('Accuracy is:', a*100)





