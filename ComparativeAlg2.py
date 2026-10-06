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
from sklearn.ensemble import GradientBoostingClassifier


# ============================
# 1. Dataset de ejemplo: crédito
# ============================

data = {
    "edad": [
        25, 28, 35, 40, 30, 45, 32, 29, 50, 38,
        27, 33, 41, 36, 31, 48, 26, 39, 42, 34
    ],
    "salario": [
        1800, 2000, 3500, 4500, 2200, 5000, 2800, 2100, 6000, 4000,
        1900, 3000, 4700, 3600, 2500, 5500, 1700, 4200, 4800, 3200
    ],
    "años_empresa": [
        1, 2, 5, 8, 2, 10, 4, 2, 15, 7,
        1, 4, 9, 6, 3, 12, 1, 8, 10, 5
    ],
    "satisfaccion": [
        2, 3, 7, 8, 3, 9, 5, 2, 9, 7,
        2, 6, 8, 6, 4, 9, 1, 7, 8, 5
    ],
    "horas_extra": [
        15, 12, 5, 4, 14, 3, 8, 13, 2, 6,
        16, 7, 4, 6, 10, 3, 18, 5, 4, 8
    ],
    "renuncia": [
        1, 1, 0, 0, 1, 0, 0, 1, 0, 0,
        1, 0, 0, 0, 1, 0, 1, 0, 0, 0
    ]
}

df = pd.DataFrame(data)

print("Dataset:")
print(df)


# ============================
# 2. Variables X e y
# ============================

X = df[[
        "edad",
        "salario",
        "años_empresa",
        "satisfaccion",
        "horas_extra"
    ]]
y = df["renuncia"]


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
    "Naive Bayes": GaussianNB(),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
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