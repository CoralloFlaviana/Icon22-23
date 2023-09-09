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

#max depth
max_depth_values = range(1,16)

train_accuracy = []
test_accuracy = []

for depth in max_depth_values:
    tree = DecisionTreeClassifier(max_depth=depth, random_state = 42)
    
    tree.fit(X_train,y_train)
    train_pred = tree.predict(X_train)
    test_pred = tree.predict(X_test)
    
    train_accuracy.append(accuracy_score(y_train, train_pred))
    test_accuracy.append(accuracy_score(y_test,test_pred))
    

plt.plot(max_depth_values, train_accuracy, label='Train Accuracy')
plt.plot(max_depth_values, test_accuracy, label='Test Accuracy')

plt.title('max_depth', fontweight = 'bold')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.xticks(range(1,16,1))

plt.show()

#min_samples_leaf
min_samples_leaf = range(5,31,5)

train_accuracy = []
test_accuracy = []

for leaf in min_samples_leaf:
    tree = DecisionTreeClassifier(min_samples_leaf=leaf,random_state = 42)
    tree.fit(X_train,y_train)
    
    train_pred = tree.predict(X_train)
    test_pred = tree.predict(X_test)
    
    train_accuracy.append(accuracy_score(y_train, train_pred))
    test_accuracy.append(accuracy_score(y_test,test_pred))
    
plt.plot(min_samples_leaf,train_accuracy, label = 'Train')
plt.plot(min_samples_leaf, test_accuracy, label = 'Test')

plt.title('min_samples_leaf', fontweight = 'bold')
plt.xlabel('min_samples_leaf')
plt.ylabel('accuracy')

plt.grid(True)
plt.legend()
plt.xticks(range(5,31,5))
plt.show()

tree = DecisionTreeClassifier(min_samples_leaf= 25,random_state = 42)

tree.fit(X_train,y_train)
accuracy = cross_val_score(tree,X_train,y_train,cv = 5, scoring = 'accuracy')

print("cross val accuracy:", accuracy)
print('\nmean accuracy = ',accuracy.mean(),'\nstd accuracy = ', accuracy.std())

y_pred = tree.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

print(classification_report(y_test, y_pred))

fig = plt.subplots(figsize = (8,5))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot = True, fmt='d', cmap='Blues')

plt.title('Confusion_matrix', fontweight = 'bold')
plt.ylabel('True Class')
plt.xlabel('Predicted class')
plt.show()