# main.py

import logging
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def guardar_predicciones_finales(resultados, nombre_archivo=None):
    """
    Guarda DataFrames de predicción en CSV.
    Puede recibir un dict de DataFrames o un DataFrame único.
    """
    os.makedirs("predict", exist_ok=True)

    if isinstance(resultados, pd.DataFrame):
        resultados_dict = {"top_k": resultados}
    else:
        resultados_dict = resultados

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rutas = {}

    for tipo, df in resultados_dict.items():
        ruta = f"predict/{nombre_archivo}_{tipo}_{timestamp}.csv"
        df.to_csv(ruta, index=False)
        rutas[tipo] = ruta

        logger.info(f"Predicciones ({tipo}) guardadas en: {ruta}")
        logger.info(f"  Columnas: {list(df.columns)}")
        logger.info(f"  Registros: {len(df):,}")
        logger.info(f"  Primeras filas:\n{df.head()}")

    return rutas


def generar_predicciones_finales(
    modelos_por_grupo: dict[str, list[lgb.Booster]],
    X_predict: pd.DataFrame,
    clientes_predict: np.ndarray,
    df_predict: pd.DataFrame,
    top_k: int = 10000,
    mes: int | None = None
) -> dict:
    os.makedirs("predict", exist_ok=True)

    logger.info("Iniciando generación de predicciones finales...")
    logger.info(f"Cantidad de clientes a predecir: {len(clientes_predict)}")
    logger.info(f"Cantidad de grupos de modelos: {len(modelos_por_grupo)}")

    predicciones_individuales = []
    resultados_ganancias = []
    preds_sum_global = np.zeros(len(X_predict), dtype=np.float32)
    n_modelos = 0
    preds_por_grupo = []

    y_true = df_predict["target"].values

    for nombre_grupo, modelos in modelos_por_grupo.items():
        logger.info(f"Procesando grupo: {nombre_grupo} con {len(modelos)} modelos")
        preds_grupo = np.zeros(len(X_predict), dtype=np.float32)

        for i, modelo in enumerate(modelos, start=1):
            y_pred_proba = modelo.predict(X_predict, num_iteration=modelo.best_iteration)
            preds_sum_global += y_pred_proba
            preds_grupo += y_pred_proba
            n_modelos += 1

            df_i = pd.DataFrame({
                "numero_de_cliente": clientes_predict,
                "probabilidad": y_pred_proba,
                "grupo": nombre_grupo,
                "modelo_id": f"{nombre_grupo}_seed{i}"
            })
            df_i["predict"] = 0
            df_i = df_i.sort_values("probabilidad", ascending=False, ignore_index=True)
            df_i.loc[:top_k - 1, "predict"] = 1

            predicciones_individuales.append(df_i)

            # Ganancia individual
            ganancia_test = ganancia_evaluator(y_pred_proba, y_true)
            resultados_ganancias.append({
                "mes": mes,
                "grupo": nombre_grupo,
                "modelo_id": f"{nombre_grupo}_seed{i}",
                "ganancia_test": float(ganancia_test)
            })

        preds_grupo /= len(modelos)
        preds_por_grupo.append(preds_grupo)

    # Guardar predicciones individuales
    if predicciones_individuales:
        df_all_preds = pd.concat(predicciones_individuales, ignore_index=True)
        df_all_preds.to_csv("predict/predicciones_individuales.csv", index=False)
        logger.info(f"CSV de predicciones individuales guardado con {len(df_all_preds)} filas")

    # Global
    y_pred_global = preds_sum_global / n_modelos
    df_topk_global = pd.DataFrame({
        "numero_de_cliente": clientes_predict,
        "probabilidad": y_pred_global
    }).sort_values("probabilidad", ascending=False, ignore_index=True)
    df_topk_global["predict"] = 0
    df_topk_global.loc[:top_k - 1, "predict"] = 1
    df_topk_global.to_csv("predict/predicciones_global.csv", index=False)

    ganancia_global = ganancia_evaluator(y_pred_global, y_true)
    resultados_ganancias.append({
        "mes": mes,
        "grupo": "GLOBAL",
        "modelo_id": "ensamble_global",
        "ganancia_test": float(ganancia_global)
    })

    # Grupos
    y_pred_grupos = sum(preds_por_grupo) / len(preds_por_grupo)
    df_topk_grupos = pd.DataFrame({
        "numero_de_cliente": clientes_predict,
        "probabilidad": y_pred_grupos
    }).sort_values("probabilidad", ascending=False, ignore_index=True)
    df_topk_grupos["predict"] = 0
    df_topk_grupos.loc[:top_k - 1, "predict"] = 1
    df_topk_grupos.to_csv("predict/predicciones_grupos.csv", index=False)

    ganancia_grupos = ganancia_evaluator(y_pred_grupos, y_true)
    resultados_ganancias.append({
        "mes": mes,
        "grupo": "GRUPOS",
        "modelo_id": "ensamble_grupos",
        "ganancia_test": float(ganancia_grupos)
    })

    df_ganancias = pd.DataFrame(resultados_ganancias)
    df_ganancias.to_csv(f"predict/ganancias_modelos_{mes}.csv", index=False)
    logger.info(f"✅ CSV de ganancias guardado: predict/ganancias_modelos_{mes}.csv")

    return {
        "top_k_global": df_topk_global,
        "top_k_grupos": df_topk_grupos,
        "ganancias": df_ganancias
    }
