# -*- coding: utf-8 -*-

from matplotlib.pylab import norm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from scipy.stats import invgamma
from tqdm import tqdm
import statsmodels.api as sm  # <--- IMPORTANTE PARA LOESS
import os
#######################################################################################
# PLOTTING FUNCTIONS
#######################################################################################

def draw_plot(resMH_sumgamma, resMH_sumr, itertot, burnin, simu_01, 
              breakpoints, simu_01_date, simu_01_mean, simu_01_sd, 
              threshold_bp, threshold_func, reconstructiontot, path, 
              style="classic", title="", showDate=False, save_fig=True):
    
    if showDate:
        try:
            idx_ = pd.to_datetime(simu_01_date)
        except:
            idx_ = np.arange(len(simu_01))
    else:
        idx_ = np.arange(len(simu_01))
    
    fig_a = plt.figure(constrained_layout=True, figsize=(20, 15), dpi=250)
    plt.style.use(style)
    gs = fig_a.add_gridspec(2, 2)

    nPlot1 = len(resMH_sumgamma)
    nPlot2 = len(resMH_sumr)
    
    # --- AX1: Breakpoints ---
    f_ax1 = fig_a.add_subplot(gs[0,0])
    
    if showDate:
        f_ax1.xaxis.set_minor_locator(ticker.MultipleLocator(115))
        f_ax1.xaxis.set_major_locator(mdates.YearLocator())
        f_ax1.xaxis.set_major_formatter(mdates.DateFormatter('\n%Y'))
        f_ax1.plot(idx_, resMH_sumgamma/(itertot-burnin), 
                   marker='o', markeredgecolor='black', markersize=3, 
                   color='white', linestyle='none')
    else:
        f_ax1.plot(np.arange(nPlot1), resMH_sumgamma/(itertot-burnin), 
                   marker='o', markeredgecolor='black', markersize=3, 
                   color='white', linestyle='none')
        
    f_ax1.axhline(y=threshold_bp, color='red', linewidth=1)
    f_ax1.set_title('Selección puntos de cambio', size=14)
    f_ax1.set_ylabel('Probabilidad posteriori', fontsize='large')

    # --- AX2: Functions ---
    f_ax2 = fig_a.add_subplot(gs[0,1])
    f_ax2.plot(np.arange(nPlot2), resMH_sumr/(itertot-burnin), 
               marker='o', markersize=3, markeredgecolor='black', 
               color='white', linestyle='none')

    f_ax2.axhline(y=threshold_func, color='red', linewidth=1)
    f_ax2.set_ylabel('Probabilidad posteriori', fontsize='large')
    f_ax2.set_title('Selección de funciones', size=14)

    # --- AX3: Reconstruction ---
    f_ax3 = fig_a.add_subplot(gs[1, :])
    
    simu_01_ = simu_01 * simu_01_sd + simu_01_mean
    reconstruction_ = reconstructiontot.reshape(-1,1) * simu_01_sd + simu_01_mean

    if showDate:
        f_ax3.plot(idx_, simu_01_, color='black', ls='solid', lw=2, label="Serie {}".format(title))
        f_ax3.plot(idx_, reconstruction_, color='#2D7AC0', lw=2, ls=(0, (5, 1)), label='Serie estimada')
        for i in breakpoints[1:]:
             if i < len(idx_): f_ax3.axvline(x=idx_[i], color='red', linewidth=2)
    else:
        f_ax3.plot(np.arange(nPlot1), simu_01_, color='grey', ls='solid', lw=2, label='Serie {}'.format(title))
        f_ax3.plot(np.arange(nPlot1), reconstruction_, color='#2D7AC0', lw=2, ls=(0, (5, 1)), label='Serie estimada')
        for i in breakpoints[1:]:
            f_ax3.axvline(x=i, color='red', linewidth=2)

    handles, labels = f_ax3.get_legend_handles_labels()
    blue_line = mlines.Line2D([],[], color='red', label='Puntos de cambio')
    handles.append(blue_line)
    f_ax3.legend(handles=handles)
    f_ax3.set_title('Serie {} y modelo estimado'.format(title), size=14)
    
    if save_fig:
        fig_a.savefig(path+'/fig_{}_result.png'.format(title), dpi=250)
        plt.close(fig_a)


def draw_article_style_plot(gamma_sums, r_sums, itertot, burnin, 
                            original_serie_norm, reconstruction_norm, 
                            detected_breakpoints, mean, sd, 
                            threshold_bp, threshold_fnc, 
                            title_suffix="", output_path='./outputs'):
    
    n_samples = itertot - burnin
    gamma_probs = gamma_sums / n_samples
    r_probs = r_sums / n_samples
    
    serie_denorm = (original_serie_norm.flatten() * sd) + mean
    reconstruction_denorm = (reconstruction_norm.flatten() * sd) + mean
    n_points = len(serie_denorm)
    time_axis = np.arange(n_points)

    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.5])

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(np.arange(len(gamma_probs)), gamma_probs, color='black', s=15, marker='o')
    ax1.axhline(y=threshold_bp, color='r', linestyle='-', linewidth=1.5, alpha=0.7)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_title('(a) Breakpoints selection', fontsize=12, fontweight='bold')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(np.arange(len(r_probs)), r_probs, color='black', s=15, marker='o')
    ax2.axhline(y=threshold_fnc, color='r', linestyle='-', linewidth=1.5, alpha=0.7)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_title('(b) Functions selection', fontsize=12, fontweight='bold')

    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(time_axis, serie_denorm, ls='solid', lw=1.5, color='grey', label='Series', alpha=0.7)
    ax3.plot(time_axis, reconstruction_denorm, '-', color='#E67E22', linewidth=2, label='Estimated expectation')
    
    bp_label_added = False
    for bp in detected_breakpoints:
        if bp < n_points:
            lbl = 'Breakpoints' if not bp_label_added else ""
            ax3.axvline(x=bp, color='red', linestyle='-', linewidth=1.5, label=lbl)
            bp_label_added = True

    ax3.set_title('(c) True and estimated expectation', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right', frameon=True)
    ax3.set_xlim(0, n_points)
    
    plt.tight_layout()
    if not os.path.exists(output_path): os.makedirs(output_path)
    plt.savefig(f"{output_path}/article_plot_{title_suffix}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

#######################################################################################
# MATH HELPERS FOR AR(1) & ROBUSTNESS
#######################################################################################

def build_Q_ar1(phi, n):
    """Construye la matriz inversa de correlación Q para un AR(1)."""
    if abs(phi) >= 0.999: phi = 0.999 * np.sign(phi)
    Q = np.zeros((n, n))
    np.fill_diagonal(Q, 1 + phi**2)
    Q[0, 0] = 1
    Q[n-1, n-1] = 1
    off_diag = -phi
    idx = np.arange(n - 1)
    Q[idx, idx + 1] = off_diag
    Q[idx + 1, idx] = off_diag
    return Q

def robust_multivariate_normal(mean, cov):
    """Generación robusta de normal multivariada."""
    try:
        return np.random.multivariate_normal(mean, cov)
    except (ValueError, RuntimeWarning):
        cov = 0.5 * (cov + cov.T) + np.eye(len(cov)) * 1e-6
        try:
            return np.random.multivariate_normal(mean, cov)
        except:
            # Fallback SVD
            u, s, vh = np.linalg.svd(cov)
            A = u @ np.diag(np.sqrt(s))
            z = np.random.normal(0, 1, len(mean))
            return mean + A @ z

##############################################################################
# SEGMENTATION ALGORITHM (Metropolis-Hastings)
##############################################################################

def segmentation_bias_MH(serie, nbiter, nburn, lec1, lec2, Fmatrix, nbSegInit,
                         nbToChangegamma, nbFuncInit, nbToChanger, Pi, eta,
                         phi=0.0, printiter=True):
    
    y = serie.reshape(-1)
    n = len(y)
    X = np.tri(n, n, 0, dtype=int)
    J = Fmatrix.shape[1]

    Q = build_Q_ar1(phi, n)

    gammamatrix = np.zeros((nbiter-nburn, n))
    rmatrix = np.zeros((nbiter-nburn, J))
    sumgamma = np.zeros(n, int) 
    nbactugamma = 0
    sumr = np.zeros(J, int)
    nbactur = 0
        
    indgamma10 = np.random.choice(np.arange(1, n), size=nbSegInit-1, replace=False)    
    gamma = np.zeros(n, int); gamma[0] = 1; gamma[indgamma10] = 1 
    indgamma1 = np.concatenate((np.array([0]), indgamma10))
    nbSeg = nbSegInit
    Xgamma = X[:, indgamma1] 

    indr10 = np.random.choice(np.arange(1, J), size=nbFuncInit-1, replace=False)
    r = np.zeros(J, int); r[0] = 1; r[indr10] = 1
    indr1 = np.concatenate((np.array([0]), indr10))
    nbFunc = nbFuncInit
    Fmatrixr = Fmatrix[:, indr1]

    def get_log_marginal_stats(X_curr, F_curr):
        if F_curr.shape[1] > 0:
            H = np.hstack([X_curr, F_curr])
            d_gam = X_curr.shape[1]
            P_inv = np.zeros((H.shape[1], H.shape[1]))
            P_inv[:d_gam, :d_gam] = (1.0/lec1) * (X_curr.T @ X_curr)
            P_inv[d_gam:, d_gam:] = (1.0/lec2) * (F_curr.T @ F_curr)
        else:
            H = X_curr
            P_inv = (1.0/lec1) * (X_curr.T @ X_curr)

        HT_Q_H = H.T @ Q @ H
        Prec_Post = HT_Q_H + P_inv
        idx_diag = np.arange(Prec_Post.shape[0])
        Prec_Post[idx_diag, idx_diag] += 1e-6

        try:
            sign, logdet_Prec = np.linalg.slogdet(Prec_Post)
            if sign <= 0: return -1e100
            
            HT_Q_y = H.T @ Q @ y
            beta_hat = np.linalg.solve(Prec_Post, HT_Q_y)
            
            res = y - H @ beta_hat
            S2 = res.T @ Q @ res + beta_hat.T @ P_inv @ beta_hat
            if S2 <= 1e-8: S2 = 1e-8

            log_ML = -0.5 * logdet_Prec - (n/2) * np.log(S2)
            return log_ML

        except np.linalg.LinAlgError:
            return -1e100

    curr_log_ML = get_log_marginal_stats(Xgamma, Fmatrixr)
    
    iterator = tqdm(range(nbiter), desc="Segmentation MH", disable=not printiter, unit="it")

    for iter in iterator:        
        choix = np.random.choice([1,2], size=1)
        
        # 1. CAMBIO EN GAMMA
        if choix == 1:
            gammaprop = gamma.copy()
            indgamma1prop = indgamma1.copy()
            nbSegprop = nbSeg
            
            indToChange = np.random.choice(np.arange(1,n), size=nbToChangegamma, replace=False)
            for i in np.arange(nbToChangegamma):
                idx = indToChange[i]
                if gamma[idx] == 0:
                    gammaprop[idx] = 1
                    indgamma1prop = np.append(indgamma1prop, idx)
                    nbSegprop += 1
                else:
                    gammaprop[idx] = 0
                    indgamma1prop = indgamma1prop[indgamma1prop != idx]
                    nbSegprop -= 1
            
            Xgammaprop = X[:, np.sort(indgamma1prop)]
            prop_log_ML = get_log_marginal_stats(Xgammaprop, Fmatrixr)
            
            log_prior_ratio = (nbSeg - nbSegprop)/2 * np.log(1+lec1) + \
                              np.sum((gammaprop-gamma)[1:] * np.log(Pi[1:]/(1-Pi[1:])))
            
            log_alpha = (prop_log_ML - curr_log_ML) + log_prior_ratio
            
            if np.log(np.random.uniform(0,1)) < min(0, log_alpha):
                gamma = gammaprop
                indgamma1 = np.sort(indgamma1prop)
                nbSeg = nbSegprop
                Xgamma = Xgammaprop
                curr_log_ML = prop_log_ML
                nbactugamma += 1
            
        # 2. CAMBIO EN R
        if choix == 2:
            rprop = r.copy()
            indr1prop = indr1.copy()
            nbFuncprop = nbFunc
            
            indToChange = np.random.choice(np.arange(1,J), size=nbToChanger, replace=False)
            for i in np.arange(nbToChanger):
                idx = indToChange[i]
                if r[idx] == 0:
                    rprop[idx] = 1
                    indr1prop = np.append(indr1prop, idx)
                    nbFuncprop += 1
                else:
                    rprop[idx] = 0
                    indr1prop = indr1prop[indr1prop != idx]
                    nbFuncprop -= 1

            Fmatrixrprop = Fmatrix[:, np.sort(indr1prop)]
            prop_log_ML = get_log_marginal_stats(Xgamma, Fmatrixrprop)
            
            log_prior_ratio = (nbFunc - nbFuncprop)/2 * np.log(1+lec2) + \
                              np.sum((rprop-r)[1:] * np.log(eta[1:]/(1-eta[1:])))

            log_alpha = (prop_log_ML - curr_log_ML) + log_prior_ratio
                        
            if np.log(np.random.uniform(0,1)) < min(0, log_alpha):
                r = rprop
                indr1 = np.sort(indr1prop)
                nbFunc = nbFuncprop
                Fmatrixr = Fmatrixrprop
                curr_log_ML = prop_log_ML
                nbactur += 1
        
        if iter >= nburn:
            sumgamma += gamma
            sumr += r
            gammamatrix[iter-nburn, :] = gamma
            rmatrix[iter-nburn, :] = r
    
    return dict(sumgamma=sumgamma, sumr=sumr, nbactugamma=nbactugamma, 
                nbactur=nbactur, gammamatrix=gammamatrix, rmatrix=rmatrix)

##############################################################################
# GIBBS SAMPLER & PHI UPDATE (STABILIZED)
##############################################################################
def update_phi_MH(resid, phi_curr, sigma2, step_size_z=0.12, a=2.0, b=2.0):
    """
    MH para phi en AR(1) con propuesta en z=atanh(phi).
    Prior opcional: Beta(a,b) sobre x=(phi+1)/2.
    """

    # clamp para evitar inf en atanh
    phi_curr = float(np.clip(phi_curr, -0.995, 0.995))
    z_curr = np.arctanh(phi_curr)

    # propuesta simétrica en z
    z_prop = np.random.normal(z_curr, step_size_z)
    phi_prop = float(np.tanh(z_prop))

    n = len(resid)

    def log_target_in_phi(phi_val):
        # Likelihood con precisión Q(phi)
        Q = build_Q_ar1(phi_val, n)
        quad = float(resid.T @ Q @ resid)
        loglik = 0.5 * np.log(1.0 - phi_val**2) - quad / (2.0 * sigma2)

        # Prior Beta en x=(phi+1)/2  (si a=b=2 es suave y evita extremos)
        x = 0.5 * (phi_val + 1.0)
        eps = 1e-12
        logprior = (a - 1.0) * np.log(x + eps) + (b - 1.0) * np.log(1.0 - x + eps)

        return loglik + logprior

    # Jacobiano por trabajar en z (phi = tanh(z))
    # target(z) = target(phi(z)) + log|dphi/dz| = target(phi) + log(1-phi^2)
    logpost_curr = log_target_in_phi(phi_curr) + np.log(1.0 - phi_curr**2 + 1e-12)
    logpost_prop = log_target_in_phi(phi_prop) + np.log(1.0 - phi_prop**2 + 1e-12)

    log_alpha = logpost_prop - logpost_curr

    if np.log(np.random.rand()) < log_alpha:
        return phi_prop, True
    return phi_curr, False

#def update_phi_MH(y_resid, phi_curr, sigma2, n, step_size=0.05):
#    """
#    Actualiza phi con un PRIOR BETA(alpha, beta) reescalado a (-1, 1).3
#    Esto penaliza los valores extremos cercanos a 1.0 para evitar que el modelo
#    se convierta en un Paseo Aleatorio y deje de detectar quiebres.
#    """
#    # Proponer nuevo valor (Random Walk)
#    phi_prop = np.random.normal(phi_curr, step_size)
#    
#    # Límite estricto de estacionariedad
#    if abs(phi_prop) >= 0.995: return phi_curr, False
#
#    def log_posterior_score(phi_val, resid, s2):
#        # 1. Log-Likelihood del AR(1)
#        Q_val = build_Q_ar1(phi_val, len(resid))
#        quad = resid.T @ Q_val @ resid
#        log_lik = 0.5 * np.log(1 - phi_val**2) - (1.0 / (2*s2)) * quad
        
        # 2. Log-Prior: Beta(5, 2) centrado en la zona positiva moderada (0.2 - 0.8)
        # Transformamos phi de (-1,1) al dominio (0,1) para usar Beta
        # x = (phi + 1) / 2
        # Prior fuerte para evitar phi > 0.9
        # Usamos una penalización simple: (1 - phi^2)^4
        # Esto empuja phi hacia 0 suavemente, castigando mucho el 1.
#        log_prior = 4 * np.log(1 - phi_val**2 + 1e-9)
        
#        return log_lik + log_prior
#
#    # Calcular Ratio de Metropolis
#    log_alpha = log_posterior_score(phi_prop, y_resid, sigma2) - \
#                log_posterior_score(phi_curr, y_resid, sigma2)
#    
#    # Decisión
#    if np.log(np.random.uniform(0, 1)) < log_alpha:
#        return phi_prop, True # Aceptar
#    else:
#        return phi_curr, False # Rechazar

def estimation_moy_biais(serie, nbiter, nburn, lec1, lec2, Fmatrix, gammahat,
                         rhat, priorminsigma2, priormaxsigma2, 
                         phi_init=0.0, step_size_z=0.12, a_phi=2.0, b_phi=2.0,
                         ignore_ar1=False, printiter=True):    
    y = serie.reshape(-1); n = len(y); X = np.tri(n, n, 0, dtype=int)    
    dgammahat = np.sum(gammahat); drhat = np.sum(rhat)
    
    phihat = 0.0 if ignore_ar1 else float(phi_init)

    sigma2hat = 1.0
    betagammahat = np.zeros(dgammahat)
    lambdarhat = np.zeros(int(drhat))
    
    # Almacenamiento de CADENAS
    chain_beta = np.zeros((dgammahat, nbiter-nburn))
    if drhat > 0: chain_lambda = np.zeros((int(drhat), nbiter-nburn))
    else: chain_lambda = None
    chain_sigma = np.zeros(nbiter-nburn)
    chain_phi = np.zeros(nbiter-nburn) # <--- CRUCIAL: CADENA DE PHI

    Xgamma = X[:, np.nonzero(gammahat==1)[0]]
    if drhat > 0: Fmatrixr = Fmatrix[:, np.nonzero(rhat==1)[0]]; FT_F = Fmatrixr.T @ Fmatrixr
    else: Fmatrixr = None; FT_F = None
    XT_X = Xgamma.T @ Xgamma
    
    iterator = range(nbiter)
    if printiter: iterator = tqdm(iterator, desc="Gibbs", leave=False)

    for iter_idx in iterator:
        # Si ignoramos AR1, Q siempre es identidad
        Q = build_Q_ar1(phihat, n) 
        
        # 1. Beta
        if drhat > 0: resid_y = y - Fmatrixr @ lambdarhat
        else: resid_y = y
        XT_Q_X = Xgamma.T @ Q @ Xgamma
        Prec_Beta = (1/sigma2hat) * (XT_Q_X + (1/lec1)*XT_X)
        Prec_Beta = 0.5*(Prec_Beta+Prec_Beta.T) + np.eye(len(Prec_Beta))*1e-6
        try: Cov_Beta = np.linalg.inv(Prec_Beta)
        except: Cov_Beta = np.eye(len(Prec_Beta))*1e-6
        Mean_Beta = Cov_Beta @ ((1/sigma2hat) * Xgamma.T @ Q @ resid_y)
        betagammahat = robust_multivariate_normal(Mean_Beta, Cov_Beta)
        
        # 2. Lambda
        if drhat > 0:
            resid_y = y - Xgamma @ betagammahat
            FT_Q_F = Fmatrixr.T @ Q @ Fmatrixr
            Prec_Lambda = (1/sigma2hat) * (FT_Q_F + (1/lec2)*FT_F)
            Prec_Lambda = 0.5*(Prec_Lambda+Prec_Lambda.T) + np.eye(len(Prec_Lambda))*1e-6
            try: Cov_Lambda = np.linalg.inv(Prec_Lambda)
            except: Cov_Lambda = np.eye(len(Prec_Lambda))*1e-6
            Mean_Lambda = Cov_Lambda @ ((1/sigma2hat) * Fmatrixr.T @ Q @ resid_y)
            lambdarhat = robust_multivariate_normal(Mean_Lambda, Cov_Lambda)

        # 3. Sigma
        if drhat > 0: mu_total = Xgamma @ betagammahat + Fmatrixr @ lambdarhat
        else: mu_total = Xgamma @ betagammahat
        resid = y - mu_total
        sse_ar1 = resid.T @ Q @ resid
        prior_pen = (1/lec1)*betagammahat.T @ XT_X @ betagammahat
        if drhat > 0: prior_pen += (1/lec2)*lambdarhat.T @ FT_F @ lambdarhat
        scale_val = 0.5 * (sse_ar1 + prior_pen)
        if scale_val <= 0: scale_val = 1e-6
        try: sigma2hat = invgamma.rvs((n + dgammahat + drhat)/2.0, scale=scale_val)
        except: sigma2hat = priorminsigma2
        
        # 4. Phi
        if not ignore_ar1:
            phihat, _ = update_phi_MH(resid, phihat, sigma2hat, step_size_z=0.12, a=2.0, b=2.0)
        
        if iter_idx >= nburn:
            idx = iter_idx - nburn
            chain_beta[:, idx] = betagammahat
            if drhat > 0: chain_lambda[:, idx] = lambdarhat
            chain_sigma[idx] = sigma2hat
            chain_phi[idx] = phihat 

    estbeta = np.mean(chain_beta, axis=1)
    estlambda = np.mean(chain_lambda, axis=1) if drhat > 0 else None
    estsigma = np.mean(chain_sigma)
    estphi = np.mean(chain_phi)
    
    # Retornamos la cadena al final
    return [chain_beta, chain_lambda, chain_sigma, estbeta, estlambda, estsigma, estphi, chain_phi]

##############################################################################
# WRAPPER FUNCTION
##############################################################################

def dbp_with_function_effect(
        Fmatrix, data_serie, itertot, burnin, 
        lec1, lec2, nbseginit, nbfuncinit, nbtochangegamma, nbtochanger,
        Pi, eta, threshold_bp, threshold_fnc, phi_init=0.0, 
        ignore_ar1=False, printiter=False, 
        fmatrixNames=None, completeSerie=None, fileName=None, ouputfolder=None):
    
    n = len(data_serie)
    phi_for_seg = 0.0 if ignore_ar1 else phi_init

    resMH = segmentation_bias_MH(
        data_serie, itertot, burnin, lec1, lec2, Fmatrix, nbseginit, 
        nbtochangegamma, nbfuncinit, nbtochanger, Pi, eta, 
        phi=phi_for_seg, printiter=printiter)
    
    probs_bp = resMH['sumgamma'] / (itertot - burnin)
    breakpoints = np.where(probs_bp > threshold_bp)[0]
    gammahat = np.zeros(n, int); gammahat[breakpoints] = 1  

    probs_fnc = resMH['sumr'] / (itertot - burnin)
    basefunctions = np.where(probs_fnc > threshold_fnc)[0]
    rhat = np.zeros(Fmatrix.shape[1]); rhat[basefunctions] = 1    
     
    estim = estimation_moy_biais(
        data_serie, itertot, burnin, lec1, lec2,
        Fmatrix, gammahat, rhat, 0.001, 5,
        phi_init=phi_for_seg, step_size_z=0.12, a_phi=2.0, b_phi=2.0,
        ignore_ar1=ignore_ar1, printiter=printiter
    )
    
    estbeta = estim[3]
    estlambda = estim[4]
    estsigma = estim[5]  # <--- CORREGIDO AQUÍ (Antes faltaba)
    estphi = estim[6]
    resphihat = estim[7] # <--- CADENA DE PHI
    
    reconstructionmu = np.zeros(n)
    if len(breakpoints) > 0:
        X_gamma_final = np.tri(n, n, 0, dtype=int)[:, breakpoints]
        reconstructionmu = X_gamma_final @ estbeta
    else: reconstructionmu[:] = np.mean(data_serie) 

    reconstructionf = np.zeros(n)
    if rhat.sum() > 0:
        F_r_final = Fmatrix[:, basefunctions]
        reconstructionf = F_r_final @ estlambda
    
    reconstructiontot = reconstructionmu + reconstructionf
    
    # Devolvemos la cadena en el índice 9 de la lista final
    return list([resMH, estim[0], estim[1], estim[2], estbeta, estlambda, estsigma, reconstructiontot, estphi, resphihat])

##############################################################################
# TWO-STAGE INIT
##############################################################################

# En functions_sim.py (Agrega esto al final o junto a las otras funciones de estimación)
def estimate_phi_local_regression(data_serie, frac=0.1):
    """
    Estima Phi usando LOESS (Locally Weighted Scatterplot Smoothing).
    frac: La fracción de datos usada para cada estimación local. 
          Un valor bajo (0.05-0.1) permite capturar cambios rápidos.
    """
    y = np.array(data_serie).flatten()
    x = np.arange(len(y))
    
    # Ajuste LOESS
    # devuelve una matriz nx2, la columna 1 son los valores ajustados
    trend_smooth = sm.nonparametric.lowess(y, x, frac=frac, return_sorted=False)
    
    # Residuos
    residuals = y - trend_smooth
    
    # Autocorrelación Lag-1
    phi_est = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
    
    # Protecciones
    if np.isnan(phi_est): phi_est = 0.5
    if phi_est > 0.99: phi_est = 0.99
    if phi_est < -0.99: phi_est = -0.99
        
    return phi_est

def estimate_phi_particle_filter(data_serie, n_particles=1000, p_jump=0.05):
    """
    Estima Phi usando un Filtro de Partículas (SMC).
    Esto es más robusto que Kalman para series con saltos abruptos (No-Gaussianos).
    """
    y = np.array(data_serie).flatten()
    n = len(y)
    
    # 1. Inicialización de Partículas
    # Cada partícula es una hipótesis: [phi, mu_actual, error_previo]
    particles_phi = np.random.uniform(0, 0.99, n_particles) # Prior uniforme para phi
    particles_mu = np.ones(n_particles) * y[0] # Inicializar en el primer dato
    particles_err = np.zeros(n_particles)
    weights = np.ones(n_particles) / n_particles
    
    # Parámetros fijos asumidos para la detección (pueden ser ajustados)
    sigma_noise = np.std(np.diff(y)) # Estimación ruda del ruido
    
    # Almacenar historia para promediar al final
    phi_history = []
    
    # 2. Bucle Temporal (Filtro)
    for t in range(1, n):
        # A. Predicción (Propagación)
        # Algunos saltan (cambio de media), otros no.
        # Máscara de saltos: 1 si salta, 0 si no
        jumps = np.random.binomial(1, p_jump, n_particles)
        
        # Si salta: Nueva media aleatoria (basada en el dato actual + ruido)
        # Si no salta: Mantiene la media anterior
        new_mu = np.where(jumps == 1, 
                          y[t] + np.random.normal(0, sigma_noise, n_particles), 
                          particles_mu)
        
        particles_mu = new_mu
        
        # Predicción del valor esperado: y_hat = mu + phi * error_previo
        y_pred = particles_mu + particles_phi * particles_err
        
        # B. Actualización (Pesos)
        # Likelihood: N(y_t | y_pred, sigma)
        lik = norm.pdf(y[t], loc=y_pred, scale=sigma_noise)
        
        # Evitar colapso numérico
        weights *= lik
        weights += 1.e-300 
        weights /= np.sum(weights) # Normalizar
        
        # C. Estimación del instante t
        phi_mean_t = np.sum(particles_phi * weights)
        phi_history.append(phi_mean_t)
        
        # D. Resampling (Solo si los pesos están muy degenerados)
        # Criterio de Efectividad de Muestra (ESS)
        eff_n = 1.0 / np.sum(weights**2)
        if eff_n < n_particles / 2.0:
            indices = np.random.choice(n_particles, size=n_particles, p=weights)
            particles_phi = particles_phi[indices]
            particles_mu = particles_mu[indices]
            particles_err = particles_err[indices]
            weights = np.ones(n_particles) / n_particles
            
            # MCMC Move (Jitter) para evitar empobrecimiento de Phi
            # Añadimos un pequeño ruido a phi para que explore
            particles_phi = particles_phi + np.random.normal(0, 0.02, n_particles)
            particles_phi = np.clip(particles_phi, -0.99, 0.99)

        # Actualizar error para el siguiente paso: e_t = y_t - mu_t
        # Nota: Usamos el mu de la partícula, no el observado
        particles_err = y[t] - particles_mu

    # La estimación final es el promedio de las estimaciones en el tiempo
    # (Opcional: tomar solo los últimos pasos si se busca convergencia)
    phi_final = np.mean(phi_history[int(n*0.2):]) # Descartar primeros 20%
    
    print(f"--- Particle Filter Estimation ---")
    print(f"Phi estimado (PF): {phi_final:.4f}")
    
    return phi_final

def estimate_phi_via_residuals(data_serie, Fmatrix, lec1_strict=50):
    """
    Estima Phi en dos etapas para evitar confundir tendencia con autocorrelación.
    
    1. Ejecuta una segmentación rápida asumiendo INDEPENDENCIA (phi=0) pero con
       una penalización ALTA (lec1_strict). Esto fuerza al modelo a capturar 
       solo los saltos estructurales grandes y ignorar el ruido AR.
    2. Calcula los residuos (Datos - Tendencia estimada).
    3. Calcula la autocorrelación de lag-1 de los residuos.
    """
    print(f"--- Pre-estimación de Phi (Método Two-Stage) ---")
    
    # Normalizar datos si no lo están
    y = np.array(data_serie).flatten()
    mean_val = np.mean(y)
    sd_val = np.std(y)
    y_norm = (y - mean_val) / sd_val
    y_norm = y_norm.reshape(-1, 1)
    
    # Configuración para segmentación "Grosera" (Strict Segmentation)
    # Usamos pocas iteraciones porque solo queremos la tendencia principal
    iter_fast = 2000 
    burn_fast = 500
    
    # Ejecutamos DBP forzando phi=0 y lec1 ALTO
    print(f"1. Segmentación estricta (Lec1={lec1_strict}, Phi=0)...")
    
    # Priors dummy
    pi_vec = np.full(len(y_norm), 0.05)
    eta_vec = np.full(Fmatrix.shape[1], 0.05)
    
    res_fast = dbp_with_function_effect(
        Fmatrix, y_norm, 
        itertot=iter_fast, burnin=burn_fast,
        lec1=lec1_strict, lec2=20, # Penalizaciones altas para evitar overfitting
        nbseginit=5, nbfuncinit=5,
        nbtochangegamma=1, nbtochanger=1,
        Pi=pi_vec, eta=eta_vec,
        threshold_bp=0.5, threshold_fnc=0.3,
        phi_init=0.0,     # <--- CLAVE: Asumir independencia
        ignore_ar1=True,  # <--- CLAVE: No estimar phi aquí
        printiter=False
    )
    
    # 2. Extraer la Tendencia Estimada (Solo saltos grandes)
    rec_norm = res_fast[7]
    trend_est = rec_norm * sd_val + mean_val
    
    # 3. Calcular Residuos
    residuals = y - trend_est
    
    # 4. Estimar Phi sobre los residuos (Correlación Lag-1)
    # Usamos np.corrcoef entre r[t] y r[t-1]
    phi_est = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
    
    # Protección
    if np.isnan(phi_est): phi_est = 0.5
    if phi_est > 0.99: phi_est = 0.99
    if phi_est < -0.99: phi_est = -0.99
        
    print(f"2. Phi estimado sobre residuos: {phi_est:.4f}")
    
    return phi_est, trend_est, residuals

##############################################################################
# SIMULATION
##############################################################################

def simuSeries(M, n, K, muChoix, varianceError, pHaar, p1, p2, phi=0.0):
    muChoix = np.array(muChoix)
    t = np.arange(n)
    Part1 = 1.5 * np.sin(2*np.pi*t/20)
    Part2 = np.zeros(n)
    for p in pHaar:
        if p < n: Part2[p] = np.random.choice([1.5, -2, 3]) 
    biais = Part1 + Part2
    
    muMat = np.full((M, n), -1.0); tauMat = np.zeros((M, K), dtype=int); erreurs = []
    # opcion llm
    for var in varianceError:
        errorMat = np.zeros((M, n))

        if not (-1.0 < phi < 1.0):
            raise ValueError("phi must be in (-1,1) for a stationary AR(1).")

        sd_u = np.sqrt(var)                      # sd de la innovación u_t
        sd_e0 = np.sqrt(var / (1.0 - phi**2))    # sd estacionaria de e_t
        for m in range(M):
            u = np.random.normal(0.0, sd_u, n)
            e = np.empty(n)
            e[0] = np.random.normal(0.0, sd_e0)  # arranque estacionario
            for i in range(1, n):
                e[i] = phi * e[i-1] + u[i]
            errorMat[m, :] = e

        erreurs.append(errorMat)
    ################
    # opcion original    
    #for var in varianceError:
    #    if not (-1.0 < phi < 1.0):
    #        raise ValueError("phi must be in (-1,1) for a stationary AR(1).")
    #    errorMat = np.zeros((M, n))
    #    sd_innov = np.sqrt(var * (1 - phi**2)) 
    #    for m in range(M):
    #        u = np.random.normal(0, sd_innov, n)
    #        e = np.zeros(n); e[0] = u[0]
    #        for i in range(1, n): e[i] = phi * e[i-1] + u[i]
    #        errorMat[m, :] = e
    #    erreurs.append(errorMat)

    for m in range(M):
        bps = np.sort(np.random.choice(np.arange(p2, n-p2), K-1, replace=False))
        bps = np.concatenate([bps, [n]])
        tauMat[m, :] = bps
        
        mutemp = np.random.choice(muChoix)
        muMat[m, :bps[0]] = mutemp
        if K > 1:
            for k in range(1, K):
                candidates = muChoix[muChoix != mutemp]
                mutemp = np.random.choice(candidates)
                muMat[m, bps[k-1]:bps[k]] = mutemp

    series_vec = np.repeat(np.arange(M), n)
    pos_vec = np.tile(np.arange(n), M)
    mu_vec = muMat.flatten()
    tau_vec = np.zeros(M*n) 
    bias_vec = np.tile(biais, M)
    err_vec = erreurs[0].flatten()
    
    df = pd.DataFrame({'series': series_vec, 'position': pos_vec, 'mu': mu_vec, 'tau': tau_vec, 'biais': bias_vec, 'erreur1': err_vec})
    return [K, muMat, tauMat, erreurs, biais, df]