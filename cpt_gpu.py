import numpy as np
import pandas as pd
from collections import namedtuple
from scipy.stats import beta, norm
from pygam import LinearGAM
from statsmodels.formula.api import ols, mnlogit

# return ResultsPartiallyConfounded(
    # *cpt(x=c, y=yhat, z=y, num_perms=num_perms, t_xy=r2_c_yhat, t_xz=r2_c_y, t_yz=r2_yhat_y, condlike_f=condlike_f,
          # mcmc_steps=mcmc_steps, return_null_dist=return_null_dist, random_state=random_state,
          # progress=progress, n_jobs=n_jobs))

# def _gauss_cdf(fit, df):
#     mu = np.array(fit.predict(df.Z))
#     resid = df.X.values - mu
#     sigma = np.repeat(np.std(resid), len(df.Z.values))
#     # X | Z = Z_i ~ N(mu[i], sig2[i])
#     return np.array([norm.logpdf(df.X.values, loc=m, scale=sigma) for m in mu]).T


# def _conditional_log_likelihood_gaussian_gam_cont_cont(X0, Z, **model_kwargs):
#     df = pd.DataFrame({
#         'Z': Z,
#         'X': X0
#     })
#     default_kwargs = {'n_splines': 8, 'dtype': ['numerical']}
#     model_kwargs = {**default_kwargs, **model_kwargs}
#     fit = LinearGAM(**model_kwargs).gridsearch(y=df.X, X=df.Z.values.reshape(-1, 1), progress=False)  # todo: multivariate case
#     return _gauss_cdf(fit, df)
#
# cond_log_lik_mat = condlike_f(x, z, **condlike_model_args)

# the original implementation was computing estimating the density of q(x|z),
# however here we are estimating the density of q(c|y)
# NOTE just to make sure the linear case works and lateron expand to the rest
def conditional_log_likelihood(x, z):
  default_kwargs = {'n_splines': 8, 'dtype': ['numerical']}
  fit = LinearGAM(**default_kwargs).gridsearch(y=x, X=z.reshape(-1, 1), progress=False)  # todo: multivariate case
  mu = np.array(fit.predict(z))
  resid = x - mu
  sigma = np.repeat(np.std(resid), len(z))
  # X | Z = Z_i ~ N(mu[i], sig2[i])
  return np.array([norm.logpdf(x, loc=m, scale=sigma) for m in mu]).T