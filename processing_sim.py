# %%
import numpy as np
from functions_sim_v2 import * 
from config_sim import * 
# import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, title
import glob
import os
import pandas as pd
from datetime import datetime

# ## carga de la serie de datos
if DEBUG: print("loading data debugging:", DEBUG)
series_files = glob.glob(os.path.join(os.getcwd() , "data/series/*.csv"))

for filename in series_files:
    a_serie = pd.read_csv(f"./data/series/{os.path.basename(filename)}", decimal='.', delimiter=',', header=None, names=["serie"])
    serie_without_nan = a_serie["serie"].dropna().to_numpy().reshape(-1, 1)  ## remove NAs
    serie_mean = serie_without_nan ### media
    serie_sd = np.std(serie_without_nan)   ## sd
    data_serie = (a_serie - serie_without_nan.mean()) / np.std(serie_without_nan)

    n = len(serie_without_nan)
    # Probabilidad de ser punto de cambio para cada punto de la serie (0.01)
    Pi = np.concatenate([np.array([1.00]), np.repeat(0.01, n-1)], axis=0)
    data_fnct = pd.read_csv(
        f'./data/basis/Fmatrix-{os.path.basename(filename)}',
        decimal='.', delimiter=','
    ) 
    Fmatrix = np.array(data_fnct)
    
    # Probabilidad que una funcion sea seleccionada
    eta = np.concatenate([[1], np.repeat(0.01, Fmatrix.shape[1]-1)])

    print(
        f'./data/series/{os.path.basename(filename)}', a_serie.shape,
        f'./data/basis/Fmatrix-{os.path.basename(filename)}', Fmatrix.shape,
        f"./data/outputs-{os.path.basename(filename)}"
    )
    
    start=datetime.now()
    if DEBUG: print(f"started at = {str(start)}")

    result_ = dbp_with_function_effect( Fmatrix, np.array(serie_without_nan), itertot, burnin, lec1, lec2,
                    nbseginit, nbfuncinit, nbtochangegamma, nbtochanger, Pi, eta,
                    threshold_bp, threshold_fnc, printiter=False,
                    fmatrixNames=list(data_fnct.columns), completeSerie=np.array(a_serie),
                    fileName=f"output-{os.path.basename(filename)}", ouputfolder="./data/outputs"
                    )
    
    crono = datetime.now() - start
    if DEBUG: print(f"ended at = { crono/60 }\n")   

    resMH = result_[0]
    # Puntos de cambio segn umbral
    breakpoints_bp = np.where(resMH["sumgamma"]/(itertot-burnin) > threshold_bp)[0]

    print("breakpoints_bp ", breakpoints_bp)
    # translations = translation(np.array(a_serie))
    # breakpoints_bp = [translations[val] for val in breakpoints_bp]
    if DEBUG: print("breakpoints_bp translated", breakpoints_bp)
# %%
# 
    resMH = result_[0]

    # Recuperar estimadores puntuales
    sigma_estimado = result_[6]
    phi_estimado = result_[8]
    
    # 1. Puntos de Cambio (Breakpoints)
    breakpoints_bp = np.where(resMH["sumgamma"]/(itertot-burnin) > threshold_bp)[0]
    
    # 2. Funciones Seleccionadas
    prob_funciones = resMH["sumr"] / (itertot - burnin)
    basefunctions_idx = np.where(prob_funciones > threshold_fnc)[0]
    
    print("\n================================================")
    print(f" RESULTADOS FINAL PARA: {os.path.basename(filename)}")
    print("================================================")
    print(f"-> Puntos de cambio (t): {breakpoints_bp}")
    print(f"-> Funciones detectadas (índices): {basefunctions_idx}")
    print(f"-> Sigma^2 Estimado: {sigma_estimado:.4f}")
    print(f"-> Phi Estimado (AR1): {phi_estimado:.4f}")
    print("================================================\n")
    
    # 3. GRAFICAR ESTILO ARTÍCULO
    reconstruction = result_[7] 
    output_folder = "./data/outputs"
    
    # Usamos el nombre del archivo y el Phi estimado para el nombre de la imagen
    titulo_archivo = f"{os.path.basename(filename).replace('.csv', '')}_Phi_{phi_estimado:.2f}"

    print(f"Generando gráfico estilo artículo en {output_folder}...")

    # LLAMADA A LA NUEVA FUNCIÓN
    draw_article_style_plot(
        gamma_sums=resMH['sumgamma'],
        r_sums=resMH['sumr'],
        itertot=itertot,
        burnin=burnin,
        original_serie_norm=data_serie.to_numpy(), # Pasamos la serie normalizada original
        reconstruction_norm=reconstruction,        # Pasamos la reconstrucción normalizada
        detected_breakpoints=breakpoints_bp,
        mean=serie_without_nan.mean(),             # Media para des-normalizar
        sd=np.std(serie_without_nan),              # SD para des-normalizar
        threshold_bp=threshold_bp,
        threshold_fnc=threshold_fnc,
        title_suffix=titulo_archivo,
        output_path=output_folder
    )
    
print("\n=== PROCESO TERMINADO ===")