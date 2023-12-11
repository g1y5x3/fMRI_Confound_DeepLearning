import numpy as np
import pandas as pd
from collections import namedtuple
from statsmodels.formula.api import ols, mnlogit
from pygam import LinearGAM

# NOTE just to make sure the linear case works and lateron expand to the rest
def conditional_log_likelihood():
  pass

def _r2_cont_cont(x, y):
    # faster than scipy or statmodels
    return np.corrcoef(x, y)[0, 1] ** 2


def _r2_cat_cont(x, y):
    df = pd.DataFrame({
        'x': x,
        'y': y
    })
    fit = ols('y ~ C(x)', data=df).fit()
    return fit.rsquared.flatten()[0]


def _r2_cont_cat(x, y):
    return _r2_cat_cont(y, x)


def _r2_cat_cat(x, y):
    df = pd.DataFrame({
        'x': x,
        'y': y
    })
    fit = mnlogit('y ~ C(x)', data=df).fit(disp=0, method='powell')
    return fit.prsquared


def _r2_cat_cat_r(x, y):
    return _r2_cat_cat(y, x)


def _r2_factory(cat_x, cat_y, reverse_cat=False):
    if cat_x and cat_y:
        if reverse_cat:
            return _r2_cat_cat_r
        else:
            return _r2_cat_cat
    elif cat_x:
        return _r2_cat_cont
    elif cat_y:
        return _r2_cont_cat
    else:
        return _r2_cont_cont


r2_y_yhat = _r2_factory(cat_y, cat_yhat, reverse_cat=True)
r2_y_c = _r2_factory(cat_y, cat_c, reverse_cat=True)
r2_yhat_c = _r2_factory(cat_yhat, cat_c, reverse_cat=True)

condlike_f = _conditional_log_likelihood_factory(cat_y, cat_c, cond_dist_method)

def _generate_X_CPT_MC(nstep, log_lik_mat, Pi, random_state=None):
    # modified version of: http: // www.stat.uchicago.edu / ~rina / cpt / Bikeshare1.html
    # Berrett, T.B., Wang, Y., Barber, R.F. and Samworth, R.J., 2020. The conditional permutation test
    # for independence while controlling for confounders.
    # Journal of the Royal Statistical Society: Series B (Statistical Methodology), 82(1), pp.175 - 197.
    print("estimation initial density")
    print(log_lik_mat.shape)
    n = len(Pi)
    npair = np.floor(n / 2).astype(int)
    rng = np.random.default_rng(random_state)
    for istep in range(nstep):
        perm = rng.choice(n, n, replace=False)
        inds_i = perm[0:npair]
        inds_j = perm[npair:(2 * npair)]
        # for each k=1,...,npair, decide whether to swap Pi[inds_i[k]] with Pi[inds_j[k]]
        log_odds = log_lik_mat[Pi[inds_i], inds_j] + log_lik_mat[Pi[inds_j], inds_i] \
                   - log_lik_mat[Pi[inds_i], inds_i] - log_lik_mat[Pi[inds_j], inds_j]
        swaps = rng.binomial(1, 1 / (1 + np.exp(-np.maximum(-500, log_odds))))
        Pi[inds_i], Pi[inds_j] = Pi[inds_i] + swaps * (Pi[inds_j] - Pi[inds_i]), Pi[inds_j] - \
                                 swaps * (Pi[inds_j] - Pi[inds_i])
    return Pi

CptResults = namedtuple('CptResults', ['r2_x_z',
                                       'r2_y_z',
                                       'r2_x_y',
                                       'expected_r2_x_y',
                                       'p',
                                       'p_ci',
                                       'null_distribution']

def cpt(x, y, z, t_xy, t_xz, t_yz, condlike_f, condlike_model_args=None, num_perms=1000, mcmc_steps=50,
        return_null_dist=False, random_state=None, progress=True, n_jobs=-1):
    if condlike_model_args is None:
        condlike_model_args = {}
    rng = np.random.default_rng(random_state)
    random_sates = rng.integers(np.iinfo(np.int32).max, size=num_perms)

    x = np.array(x)

    r2_x_z = t_xz(x, z)
    r2_y_z = t_yz(y, z)
    r2_x_y = t_xy(x, y)

    cond_log_lik_mat = condlike_f(x, z, **condlike_model_args)
    Pi_init = _generate_X_CPT_MC(mcmc_steps * 5, cond_log_lik_mat, np.arange(len(x), dtype=int),
                                 random_state=random_state)

    def workhorse(_random_state):
        # batched os job_batch for efficient parallelization
        Pi = _generate_X_CPT_MC(mcmc_steps, cond_log_lik_mat, Pi_init, random_state=_random_state)
        return t_xy(x[Pi], y)

    with tqdm_joblib(tqdm(desc='Permuting', total=num_perms, disable=not progress)):
        r2_xpi_y = np.array(Parallel(n_jobs=n_jobs)(delayed(workhorse)(i) for i in random_sates))

    expected_x_y = np.quantile(r2_xpi_y, (0.05, 0.5, 0.95))
    p = np.sum(r2_xpi_y >= r2_x_y) / len(r2_xpi_y)
    ci_p = _binom_ci(len(r2_xpi_y) * p, len(r2_xpi_y))

    if not return_null_dist:
        r2_xpi_y = None

    return CptResults(
        r2_x_z,
        r2_y_z,
        r2_x_y,
        expected_x_y,
        p,
        ci_p,
        r2_xpi_y
    )
Pi