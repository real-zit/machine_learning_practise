#machine learning example for using various algorithm in finding predicting whether the note is fake or not.
import pandas as pd
import numpy as np
from sklearn.utils import shuffle

#importing the dataset
df = pd.read_csv('banknotes.csv')

#shuffling the dataset
df = shuffle(df)


X = df.iloc[:, :4].values
y = df.iloc[:, -1].values

from sklearn.preprocessing import StandardScaler
scaller = StandardScaler()
X = scaller.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train , X_test, y_train, y_test = train_test_split(X, y , test_size = 0.2, random_state = 0)


#importing the LogisticRegression
from sklearn.linear_model  import LogisticRegression
model = LogisticRegression()
model.fit(X_train , y_train)
q= model.score(X_test, y_test)

#KNeighborClassifier 
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier()
kn.fit(X_train , y_train)
kn_score=kn.score(X_test, y_test)
y_pred= kn.predict(X_test)

#creating the confusion matrix which is not necessary as we have used score method earlier
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier()
model.fit(X_train , y_train)
score2 = model.score(X_test, y_test)