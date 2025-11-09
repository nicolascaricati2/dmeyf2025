import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import os
from config import archivo_base

# Cargar el estudio
study = optuna.load_study(
    study_name=archivo_base,
    storage=f"sqlite:///../../../buckets/b1/Compe_02/optuna_db/{archivo_base}.db"
)

# Crear carpeta de resultados
os.makedirs("resultados", exist_ok=True)

# 1. Histograma de ganancia
def plot_histograma_ganancia(study):
    valores = [t.value for t in study.trials if t.value is not None]
    plt.figure(figsize=(8, 4))
    sns.histplot(valores, bins=20, kde=True)
    plt.title("Distribución de ganancia en validación")
    plt.xlabel("Ganancia")
    plt.ylabel("Frecuencia")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"resultados/{archivo_base}_histograma_ganancia.png")
    plt.close()

# 2. Evolución de ganancia por trial
def plot_ganancia_por_trial(study):
    valores = [t.value for t in study.trials if t.value is not None]
    plt.figure(figsize=(8, 4))
    plt.plot(valores, marker='o')
    plt.title("Ganancia por número de trial")
    plt.xlabel("Trial")
    plt.ylabel("Ganancia")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"resultados/{archivo_base}_ganancia_por_trial.png")
    plt.close()

# 3. Evolución del mejor valor
def plot_mejor_valor(study):
    mejores_valores = []
    mejor_actual = float('-inf')
    for t in study.trials:
        if t.value is not None:
            mejor_actual = max(mejor_actual, t.value)
            mejores_valores.append(mejor_actual)
    plt.figure(figsize=(8, 4))
    plt.plot(mejores_valores, marker='o')
    plt.title("Evolución del mejor valor de ganancia")
    plt.xlabel("Trial")
    plt.ylabel("Mejor ganancia acumulada")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"resultados/{archivo_base}_mejor_valor.png")
    plt.close()

# 4. Importancia de hiperparámetros
def plot_importancia_hiperparametros(study):
    fig = optuna.visualization.matplotlib.plot_param_importances(study)
    fig.figure.set_size_inches(8, 4)
    fig.figure.tight_layout()
    fig.figure.savefig(f"resultados/{archivo_base}_importancia_hiperparametros.png")
    plt.close()

# 5. Dispersión hiperparámetro vs ganancia
def plot_param_vs_ganancia(study, param_name):
    valores = []
    params = []
    for t in study.trials:
        if t.value is not None and param_name in t.params:
            valores.append(t.value)
            params.append(t.params[param_name])
    plt.figure(figsize=(8, 4))
    plt.scatter(params, valores, alpha=0.7)
    plt.title(f"{param_name} vs Ganancia")
    plt.xlabel(param_name)
    plt.ylabel("Ganancia")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"resultados/{archivo_base}_{param_name}_vs_ganancia.png")
    plt.close()

# Ejecutar todos los gráficos
plot_histograma_ganancia(study)
plot_ganancia_por_trial(study)
plot_mejor_valor(study)
plot_importancia_hiperparametros(study)
for param in ["learning_rate", "num_leaves", "feature_fraction", "bagging_fraction", "min_data_in_leaf"]:
    plot_param_vs_ganancia(study, param)

# Exportar resultados como JSON y CSV
datos_trials = []
for t in study.trials:
    if t.value is not None:
        fila = {
            "trial_number": t.number,
            "ganancia": t.value,
            **t.params
        }
        datos_trials.append(fila)

# Guardar JSON
with open(f"resultados/{archivo_base}_iteraciones.json", "w") as f:
    json.dump(datos_trials, f, indent=2)

# Guardar CSV
df_trials = pd.DataFrame(datos_trials)
df_trials.to_csv(f"resultados/{archivo_base}_iteraciones.csv", index=False)

print(f"✅ Todo guardado en carpeta 'resultados/' con base '{archivo_base}'")
