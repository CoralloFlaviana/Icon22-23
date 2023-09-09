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

X = df.drop('output', axis = 1)
y = df['output']

# Prepara i dati e dividi il dataset in set di addestramento e test
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.1, shuffle=True, random_state = 42)


n_estimators_values = range(100, 1001, 100)


max_depth_values = range(1, 11)


fig, axes = plt.subplots(5,2, figsize=(15, 30))

for i, n_estimators in enumerate(n_estimators_values):
    train_accuracy = []
    test_accuracy = []

    for max_depth in max_depth_values:
        rfc = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

        rfc.fit(X_train, y_train)
        train_pred = rfc.predict(X_train)
        test_pred = rfc.predict(X_test)

        train_accuracy.append(accuracy_score(y_train, train_pred))
        test_accuracy.append(accuracy_score(y_test, test_pred))

    # Создание отдельного графика на своей оси (axes)
    ax = axes[i//2, i%2]

    ax.plot(max_depth_values, train_accuracy, label=f'Train Accuracy, n_estimators={n_estimators}')
    ax.plot(max_depth_values, test_accuracy, label=f'Test Accuracy, n_estimators={n_estimators}')

    ax.set_title(f'n_estimators={n_estimators}', fontweight='bold')
    ax.set_xlabel('Max Depth')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True)
    ax.set_xticks(range(1, 11, 1))

plt.tight_layout()
plt.show()

rfc = RandomForestClassifier(max_depth = 1,
                             n_estimators = 500,
                             random_state = 42)

rfc.fit(X_train,y_train)

accuracy = cross_val_score(rfc, X_train, y_train, cv = 5, scoring = 'accuracy')

print(accuracy)
print('\nmean accuracy', accuracy.mean(),'\nstd accuracy', accuracy.std())

y_pred = rfc.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

print(classification_report(y_test,y_pred))

fig = plt.subplots(figsize = (8,5))

cm = confusion_matrix(y_test,y_pred)

sns.heatmap(cm, annot = True, fmt='d', cmap='Blues')

plt.title('Confusion_matrix', fontweight = 'bold')
plt.ylabel('True Class')
plt.xlabel('Predicted class')
plt.show()

