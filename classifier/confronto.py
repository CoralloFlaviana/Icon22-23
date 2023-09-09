import matplotlib.pyplot as plt

# Nomi dei modelli
modelli = ['Decision Tree', 'Random Forest', 'Ada Boost', 'XGBoost', 'kNN']

# Accuracy dei modelli
accuracies = [87, 90, 77, 77, 93]

# Creazione del grafico a barre
plt.figure(figsize=(10, 6))
plt.barh(modelli, accuracies, color='skyblue')
plt.xlabel('Accuracy')
plt.title('Confronto tra Modelli di Classificazione')
plt.xlim(0, 100)

# Aggiungi valori sopra le barre
for i, v in enumerate(accuracies):
    plt.text(v + 1, i, str(v) + '%', va='center', color='black', fontweight='bold')

# Visualizza il grafico
plt.show()
