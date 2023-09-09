import pandas as pd
import ydata_profiling
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
import seaborn as sns 

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import fbeta_score

from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("icon\dataset\heart.csv")
df.head()

#cancello duplicati
df = df.drop_duplicates()

scaler = StandardScaler()

df_scaler = df.copy()
df_scaler.head()

num = ['age','trtbps','chol','thalachh','oldpeak']

scaled = scaler.fit_transform(df_scaler[num])
df_scaler[num] = scaled

df_scaler.head()

X = df.drop('output', axis = 1)
y = df['output']

# Prepara i dati e dividi il dataset in set di addestramento e test
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.1, shuffle=True, random_state = 42)

knn = KNeighborsClassifier()

knn.fit(X_train,y_train)

accuracy = cross_val_score(knn, X_train, y_train, cv = 5, scoring = 'accuracy')

print(accuracy)
print('\nmean =', accuracy.mean(),'\nstd =', accuracy.std())

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

knn = KNeighborsClassifier(n_neighbors=29)

knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

print(classification_report(y_test, y_pred))

