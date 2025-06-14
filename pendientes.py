import pandas as pd
import numpy as np
from joblib import Parallel, delayed

# --- Cálculo de pendientes, medias ponderadas y medianas por grupo ---
def calcular_pendientes_grupo(group, periodos_list):
    group = group.sort_values(by='PERIODO').copy()
    n = len(group)
    y_values = group['TN'].values
    for cant in periodos_list:
        col_pend = f'PENDIENTE_TENDENCIA_{cant}'
        col_ewma = f'TN_EWMA_{str(cant).zfill(2)}'
        col_median = f'TN_MEDIAN_{str(cant).zfill(2)}'
        col_minimo = f'TN_MIN_{str(cant).zfill(2)}'
        col_maximo = f'TN_MAX_{str(cant).zfill(2)}'
        pendientes = np.full(n, np.nan)
        ewmas = np.full(n, np.nan)
        medians = np.full(n, np.nan)
        minimo = np.full(n, np.nan)
        maximo = np.full(n, np.nan)
        x = np.arange(cant)
        if n >= cant:
            for i in range(cant - 1, n):
                y = y_values[i - (cant - 1): i + 1]
                pendientes[i] = np.polyfit(x, y, 1)[0]
                ewmas[i] = pd.Series(y).ewm(span=cant, adjust=False).mean().iloc[-1]
                medians[i] = np.median(y)
                minimo[i] = np.min(y)
                maximo[i] = np.max(y)
        group[col_pend] = pendientes
        group[col_ewma] = ewmas
        group[col_median] = medians
        group[col_minimo] = minimo
        group[col_maximo] = maximo
    return group

# --- Paralelización con joblib ---
def calcular_pendientes_parallel(df, periodos_list=[6, 12], n_jobs=20):
    df = df[['CUSTOMER_ID', 'PRODUCT_ID', 'PERIODO', 'TN']].copy()
    grupos = [group for _, group in df.groupby(['PRODUCT_ID', 'CUSTOMER_ID'])]
    resultados = Parallel(n_jobs=n_jobs, backend='loky', verbose=10)(
        delayed(calcular_pendientes_grupo)(group, periodos_list) for group in grupos
    )
    df_final = pd.concat(resultados, ignore_index=True)
    return df_final

# --- Script principal ---
if __name__ == "__main__":
    # Cargar solo columnas necesarias
    df = pd.read_parquet('./data/l_vm_completa_train.parquet', engine='fastparquet', columns=[
        'CUSTOMER_ID', 'PRODUCT_ID', 'PERIODO', 'TN'
    ])

    # Calcular pendientes, medias ponderadas y medianas para los periodos indicados en paralelo
    df_resultado = calcular_pendientes_parallel(df, periodos_list=[3, 6, 9, 12], n_jobs=20)

    # Eliminar la columna TN si no la necesitas
    df_resultado.drop(columns=['TN'], inplace=True)

    # Guardar resultado
    df_resultado.to_parquet('./data/l_vm_completa_train_pendientes.parquet', engine='fastparquet', index=False)
    print("Archivo parquet guardado en ./data/l_vm_completa_train_pendientes.parquet")
