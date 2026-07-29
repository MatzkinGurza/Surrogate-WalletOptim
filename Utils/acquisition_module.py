import numpy as np
from scipy.stats import norm, qmc
from scipy.optimize import minimize
from scipy.linalg import helmert


def _sobol_simplex(n_assets, n, seed=0):
    """
    Draws n Sobol portfolios on the simplex (K x n_assets, rows sum to 1).
    Uses the sorted-cuts map so points are uniform over the simplex.
    """
    dim = n_assets - 1
    eng = qmc.Sobol(d=dim, scramble=True, seed=seed)
    cuts = eng.random(n)
    pad = np.empty((n, dim + 2))
    pad[:, 0] = 0.0
    pad[:, -1] = 1.0
    pad[:, 1:-1] = np.sort(cuts, axis=1)
    return np.diff(pad, axis=1)


def expected_improvement(mu, sigma, best, maximize=True, xi=0.01):
    """
    Expected Improvement at points with posterior mean mu and std sigma.
    best: incumbent value (best true measurement seen so far).
    maximize: True to seek larger measure values, False for smaller.
    xi: exploration margin; larger favors exploring uncertain regions.
    Returns the EI score per point (>= 0); higher means more promising.
    """
    sigma = np.maximum(sigma, 1e-12)
    if maximize:
        imp = mu - best - xi
    else:
        imp = best - mu - xi
    z = imp / sigma
    ei = imp * norm.cdf(z) + sigma * norm.pdf(z)
    ei[sigma < 1e-12] = 0.0
    return np.maximum(ei, 0.0)


class AcquisitionEI:
    """
    Expected-Improvement acquisition over the simplex for a fitted SurrogateGPR.

    surrogate: a fitted SurrogateGPR instance.
    n_assets: number of assets (simplex dimension).
    maximize: True to maximize the measure (Sharpe, STARR), False to minimize it
        (CVaR, variance). Passed per call to propose(), defaulting to this.
    xi: exploration margin for EI.
    n_candidates: number of Sobol candidates in the dense global search.
    refine: if True, polish the best candidate with a continuous optimizer.
    random_state: seed for the Sobol candidate draws.
    """

    def __init__(self, surrogate, n_assets, maximize=True, xi=0.01,
                 n_candidates=4096, refine=True, random_state=0):
        self.surrogate = surrogate
        self.n_assets = n_assets
        self.maximize = maximize
        self.xi = xi
        self.n_candidates = n_candidates
        self.refine = refine
        self.random_state = random_state
        self._U = helmert(n_assets)

    def _score(self, W, best, maximize):
        """EI score for portfolios W (K x n_assets) given the incumbent best."""
        mu, sigma = self.surrogate.predict(W, return_std=True)
        return expected_improvement(mu, sigma, best, maximize, self.xi)

    def _incumbent(self, y_observed, maximize):
        """Best true value seen so far, respecting the optimization direction."""
        return np.max(y_observed) if maximize else np.min(y_observed)

    def _refine_from(self, w0, best, maximize):
        """
        Continuous refinement of a single portfolio, run in the Helmert-projected
        space so the simplex constraint sum(w)=1 is automatic. Returns the refined
        portfolio weights, or w0 if refinement leaves the simplex or fails.
        """
        x0 = w0 @ self._U.T

        def neg_ei(x):
            w = x @ self._U + 1.0 / self.n_assets
            if np.any(w < -1e-9):
                return 1e6
            w = np.clip(w, 0, None)
            w = w / w.sum()
            val = self._score(w[None, :], best, maximize)[0]
            return -val

        res = minimize(neg_ei, x0, method="Nelder-Mead",
                       options={"xatol": 1e-4, "fatol": 1e-8, "maxiter": 200})
        w = res.x @ self._U + 1.0 / self.n_assets
        if np.any(w < -1e-9):
            return w0
        w = np.clip(w, 0, None)
        return w / w.sum()

    def propose(self, y_observed, maximize=None):
        """
        Proposes the next portfolio to evaluate.
        y_observed: array of true measure values already collected (the surrogate's
            training targets); used to set the incumbent best.
        maximize: overrides the instance default for this call if given.
        Returns a tuple (w_next, ei_value):
          w_next   : proposed portfolio weights (length n_assets, sums to 1)
          ei_value : the EI score at that point
        """
        maximize = self.maximize if maximize is None else maximize
        best = self._incumbent(np.asarray(y_observed), maximize)

        cand = _sobol_simplex(self.n_assets, self.n_candidates, self.random_state)
        scores = self._score(cand, best, maximize)
        i = int(np.argmax(scores))
        w_best, ei_best = cand[i], scores[i]

        if self.refine:
            w_ref = self._refine_from(w_best, best, maximize)
            ei_ref = self._score(w_ref[None, :], best, maximize)[0]
            if ei_ref >= ei_best:
                w_best, ei_best = w_ref, ei_ref

        return w_best, float(ei_best)