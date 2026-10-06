import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


# ============================
# 1. Dataset de ejemplo: crédito
# ============================

data = {
    "edad": [25, 30, 35, 40, 45, 50, 23, 28, 38, 42, 55, 48, 33, 29, 60],
    "ingreso": [2000, 2500, 3000, 3500, 4000, 4500, 1800, 2200, 3200, 3900, 5000, 4700, 2800, 2400, 5500],
    "historial_crediticio": [0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1],
    "deuda_actual": [1500, 1200, 800, 700, 600, 500, 1800, 1600, 900, 750, 400, 450, 1300, 1400, 300],
    "aprobado": [0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1]
}

df = pd.DataFrame(data)

print("Dataset:")
print(df)


# ============================
# 2. Variables X e y
# ============================

X = df[["edad", "ingreso", "historial_crediticio", "deuda_actual"]]
y = df["aprobado"]


# ============================
# 3. División entrenamiento/prueba
# ============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)


# ============================
# 4. Modelos a comparar
# ============================

modelos = {
    "Regresión Logística": LogisticRegression(max_iter=1000),
    "Árbol de Decisión": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "SVM": SVC(kernel="linear"),
    "Naive Bayes": GaussianNB()
}


# ============================
# 5. Entrenamiento y evaluación
# ============================

resultados = []

for nombre, modelo in modelos.items():
    print("\n" + "="*60)
    print("Modelo:", nombre)
    print("="*60)

    # Entrenar
    modelo.fit(X_train, y_train)

    # Predecir
    y_pred = modelo.predict(X_test)

    # Métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # Validación cruzada
    cv_scores = cross_val_score(modelo, X, y, cv=5, scoring="accuracy")
    cv_mean = cv_scores.mean()

    # Guardar resultados
    resultados.append({
        "Modelo": nombre,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "CV Accuracy Promedio": cv_mean
    })

    # Mostrar matriz de confusión
    print("\nMatriz de confusión:")
    print(confusion_matrix(y_test, y_pred))

    # Mostrar reporte
    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Mostrar validación cruzada
    print("Validación cruzada:", cv_scores)
    print("Promedio validación cruzada:", cv_mean)


# ============================
# 6. Tabla comparativa final
# ============================

df_resultados = pd.DataFrame(resultados)

print("\n" + "="*60)
print("TABLA COMPARATIVA DE MODELOS")
print("="*60)

print(df_resultados.sort_values(by="F1-score", ascending=False))


# ============================
# 7. Seleccionar mejor modelo
# ============================

mejor_modelo = df_resultados.sort_values(by="F1-score", ascending=False).iloc[0]

print("\nMejor modelo según F1-score:")
print(mejor_modelo)