import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import silhouette_score


# Carica il dataset
data = pd.read_csv("icon\dataset\heart.csv")

# applicazione del one-hot encoding sulle feature di tipo stringa
categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exng', 'slp', 'caa', 'thall']

onehot = OneHotEncoder(sparse_output=False)
encoded_data = onehot.fit_transform(data[categorical_features])

# combinazione delle feature encodate con il resto del dataset
pd.concat([data.drop(columns=categorical_features), pd.DataFrame(encoded_data)], axis=1)

#individuo k ottimale col metodo del gomito
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_) 
    
plt.plot(range(1, 11), wcss, 'bx-')
plt.title('elbow method')
plt.xlabel('Numero di cluster (K)')
plt.ylabel('WCSS')
plt.show()

#trovato k come ottimale
kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
kmeans.fit(data)

#somma dei quadrati intra-cluster (WCSS)
wcss = kmeans.inertia_
print("WCSS:", wcss)

#Silhouette Score
silhouette_avg = silhouette_score(data, kmeans.labels_)
print("silhouette score:", silhouette_avg)

#aggiungo label cluster
data['cluster'] = kmeans.labels_

