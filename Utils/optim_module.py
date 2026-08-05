import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Literal, Optional, Tuple, Union, Any, Type
from dataclasses import dataclass, field

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel, DotProduct, RBF, RationalQuadratic
from sklearn.gaussian_process.kernels import (
    Matern,           # o componente de resíduo local suave
    ConstantKernel,   # veículo da amplitude σ_f (a que multiplica o Matérn)
    WhiteKernel,      # ruído de estimativa
    DotProduct,       # base para tendência polinomial
    RBF,              # kernel simples infinitamente diferenciável - funções exageradamente lisas
    RationalQuadratic # mistura de RBFs em multiplas escalas
)
from sklearn.gaussian_process.kernels import Kernel
from scipy.linalg import helmert

import warnings
from typing import Optional

import numpy as np
from scipy.linalg import helmert
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel


from Utils.surrogate_module import SurrogateGPR, SurrogateKernels
import autograd.numpy as anp
from autograd import grad as _agrad
from tqdm.auto import tqdm


kernel_type = Union[SurrogateKernels.Markowitz, 
                    SurrogateKernels.Plain, 
                    SurrogateKernels.Tail, 
                    SurrogateKernels.TwoScaleMatern, 
                    SurrogateKernels.MultiScale]


class EFOptimizer:
    """
    Efficient Frontier Optimizer class for Gaussian Process Regressor and a flexible kernel class.
    Method: 
    Optimizes through gradient descent working with the fitted parameters of the GPR and its kernel.
    Descent is restricted to simplex by running on the corresponding Helmert space intrinsic to the SurrogatModel, is limited to positiv values and is regularized for portfolio returns restriction.
    Regularization is done through the augmented Lagrangian method (method of multipliers): the return
    constraint R op G (op in {=, >=}) is enforced by a Lagrange multiplier updated between rounds of
    descent, plus a moderate fixed-ish quadratic penalty rho, which avoids the manual, ill-conditioned
    tuning of a single penalty coefficient that a pure penalty method would require.
    Uses autograd to compute gradients directly related to the kernel hyperparameters and the GPR fitted parameters.
    For a given G level of return, the optimizer minimizes the risk function approximated by the GPR model and its kernel, while ensuring that the portfolio weights sum to 1 and are non-negative.
    Optimizer can adapt for G restirction as inequality or equality constraint, as well as no G constraint on returns.
    Optimizer uses n_restarts to avoid local minima and find the best solution for the given G level of return.
    The fitted target can come in either sign convention: a non-negative magnitude where smaller is
    safer (e.g. Markowitz variance), or a signed quantity where a higher raw value is safer (e.g. this
    project's CVaR, a signed tail return where less negative is safer, or a ratio like STARR/Sharpe
    where higher is always better). higher_is_better tells the optimizer which one it is dealing with,
    so gradient descent, the augmented-Lagrangian objective, and multi-restart selection all
    consistently move towards genuinely lower risk regardless of that convention.
    """
    def __init__(self, surrogate: SurrogateGPR, kernel_cls: kernel_type, higher_is_better: bool,
                 n_restarts: int = 10):
        """
        Initializes the EFOptimizer with a Gaussian Process model and a kernel class.
        higher_is_better: whether a higher raw value of the fitted target (surrogate.predict /
            self.surface) means lower risk. True for this project's CVaR (less negative is safer)
            and for ratio measures like STARR/Sharpe (higher is always better). False for a measure
            that is already a non-negative loss/dispersion magnitude where smaller is safer, such as
            Markowitz variance. Required, with no default: the surrogate's own sim_obj label does not
            by itself say which convention it uses, so the direction has to be a deliberate choice
            here rather than a silently wrong assumption.
        """
        # building blocks
        self.surrogate = surrogate
        self.ndims = surrogate.n_assets - 1
        self.gp_ = surrogate.gp_
        self.kernel_cls = kernel_cls
        self.kernel_ = self.gp_.kernel_
        self.params_ = self.gp_.kernel_.get_params()
        self.hp_ = kernel_cls._read_params(kernel_origin= surrogate.kernel_obj, p=self.params_, ndims=self.ndims)
        self.n_restarts = n_restarts
        self.higher_is_better = higher_is_better
        # cost_sign turns the raw fitted target into a quantity where lower always means safer:
        # gradient descent, the augmented-Lagrangian objective, and multi-restart selection all work
        # with cost_sign * surface(x_h) instead of surface(x_h) directly.
        self._cost_sign = -1.0 if higher_is_better else 1.0
        self._U = surrogate._U  # Helmert matrix for mapping to simplex
        # constants
        self._Xtr    = self.gp_.X_train_          # (N, n_dims) já no espaço de Helmert
        self._alpha  = self.gp_.alpha_.ravel()    # (N,) coeficientes duais
        self._y_std  = self.gp_._y_train_std      # normalize_y: desvio
        self._y_mean = self.gp_._y_train_mean     # normalize_y: média
        # automatic gradient from surface with autograd on x_h (helmert space)
        self._surface_grad = _agrad(self.surface) # gradient of the surface function with respect to x_h (helmert space coordinates)

    def surface(self, x_h):
        """
        GPR posterior average on x_h point from helmert space.
        Closed formula is differentiable: y_std * (alpha . k(x_h, X_train)) + y_mean.
        The eval_autograd of the kernel reconstructs only the signal terms (without white noise),
        which coincides with the mean outside the training points.
        """
        k_vec = self.kernel_cls.eval_autograd(x_h, self._Xtr, self.hp_)
        return self._y_std * anp.dot(self._alpha, k_vec) + self._y_mean

    def surface_grad(self, x_h):
        """
        Surface gradient in relation to x_h (Helmert space), via autograd.
        """
        return self._surface_grad(anp.asarray(x_h, dtype=float))

    def _simplex_to_helmert(self, w):
        """
        Takes weights in the simplex (n_assets,) and returns the corresponding point in Helmert space (n_assets-1,).
        """
        return anp.dot(self._U, w)

    def _helmert_to_simplex(self, x_h):
        """
        Takes a point in Helmert space (n_assets-1,) and returns the corresponding weights in the simplex (n_assets,).
        The transformation is given by w = U^T x_h + 1/n, where U is the Helmert matrix and n is the number of assets.
        """
        n = self._U.shape[1]
        return anp.dot(self._U.T, x_h) + anp.ones(n) / n

    def _penalty_func(self, x_h, mu: np.ndarray, G:float, penalty_type: Literal["eq", "ineq"]):
        """
        Penalty function for the G returns constraint.
        mu: expected returns per asset (n_assets,).
        G:  Restriction of expected return.
        mode: "eq" para igualdade, "ineq" para R >= G.
        If penalty_type is "eq", the penalty is 0.5*(G - R)^2, where R is the expected return of the portfolio given by mu^T w, and w is the weights in the simplex corresponding to x_h. 
        If penalty_type is "ineq", the penalty is 0.5*max(0, G - R)^2, where R is the expected return of the portfolio given by mu^T w, and w is the weights in the simplex corresponding to x_h.
        """
        w = self._helmert_to_simplex(x_h)
        R = np.dot(np.asarray(mu, dtype=float), w) 
        delta = G - R
        if penalty_type == "eq":
            return 0.5 * delta**2
        elif penalty_type == "ineq":
            return 0.5 * max(0, delta)**2
        else:
            raise ValueError("penalty_type must be 'equality' or 'inequality'")

    
    def penalty_grad(self, x_h, mu: np.ndarray, G: float, penalty_type: Literal["eq", "ineq"],
                      lam: float, rho: float):
        """
        Gradient, with respect to x_h (Helmert space), of the augmented-Lagrangian term for the
        G returns constraint (method of multipliers, not a pure quadratic penalty).
        The return R(x_h) is affine in Helmert space, R(x_h) = (U mu) . x_h + (1/n) sum(mu), so its
        gradient is the constant vector c = U mu, independent of x_h.
        For L_rho(x_h, lam) = f(x_h) - lam*(R - G) + (rho/2)*(R - G)^2 (equality case), the gradient
        of the constraint term in x_h is (-lam + rho*(R - G)) * c.
        For the inequality case (R >= G), the multiplier is kept non-negative via the projected
        update lam <- max(0, lam - rho*(R - G)) (done by the caller, between rounds); consistently,
        the gradient factor here uses the *projected* multiplier max(0, lam - rho*(R - G)), which is
        exactly zero unless the constraint is currently violated (R < G) or was violated recently
        enough that lam is still positive -- i.e. proper KKT complementarity, not a naive "only act on
        violated steps" clip.
        Parameters:
        x_h: point in Helmert space (n_assets-1,)
        mu: expected returns per asset (n_assets,)
        G:  target return level.
        penalty_type: "eq" for equality, "ineq" for R >= G.
        lam: current Lagrange multiplier estimate (held fixed within the inner descent).
        rho: current penalty coefficient (held fixed within the inner descent).
        Returns the gradient (factor * c) and the value of R (useful for monitoring violation).
        """
        c = self._U @ np.asarray(mu, dtype=float)       # U mu, shape (n-1,)
        n = self._U.shape[1]
        b = np.sum(mu) / n
        R = c @ np.asarray(x_h, dtype=float) + b        # retorno da carteira em x_h
        delta = R - G
        if penalty_type == "eq":
            factor = -lam + rho * delta
        elif penalty_type == "ineq":
            factor = -max(0.0, lam - rho * delta)
        else:
            raise ValueError("penalty_type must be 'eq' or 'ineq'")
        return factor * c, R


    def _objective_grad(self, x_h, mu: np.ndarray, G: float, penalty_type: Literal["eq", "ineq"],
                         lam: float, rho: float):
        """
        Computes the gradient of the augmented-Lagrangian objective with respect to x_h (Helmert space).
        The objective is cost_sign * the GPR surface (see __init__'s higher_is_better) plus the
        augmented-Lagrangian term for the G returns constraint, evaluated with the current multiplier
        lam and penalty coefficient rho held fixed (both are only updated by the outer loop, between
        rounds of this inner objective).
        Parameters:
        x_h: point in Helmert space (n_assets-1,)
        mu: expected returns per asset (n_assets,)
        G:  target return level.
        penalty_type: "eq" for equality, "ineq" for R >= G.
        lam: current Lagrange multiplier estimate.
        rho: current penalty coefficient.
        Returns the gradient of the objective function with respect to x_h (n_assets-1,) and the expected return R.
        """
        grad_surface = self._cost_sign * self.surface_grad(x_h)  # gradient of the cost-oriented GPR surface
        grad_penalty, R = self.penalty_grad(x_h, mu, G, penalty_type, lam, rho)  # augmented-Lagrangian term
        return grad_surface + grad_penalty, R

    @staticmethod
    def _project_simplex(v: np.ndarray) -> np.ndarray:
        """
        Euclidean projection of v onto the probability simplex {w: w >= 0, sum(w) = 1}.
        Implements the exact algorithm of Wang & Carreira-Perpinan (2013): sort v in
        descending order, form the running sums-minus-one, find the largest index rho_idx
        for which u[rho_idx] - (cumsum[rho_idx] - 1) / (rho_idx + 1) is still positive, set
        tau = (cumsum[rho_idx] - 1) / (rho_idx + 1) and return max(v - tau, 0).
        This is the closest point on the simplex to v; unlike "clip negatives and
        renormalize", it is unbiased and exact.
        v: array (n_assets,), arbitrary real values.
        Returns the projected weights (n_assets,).
        """
        v = np.asarray(v, dtype=float)
        n = v.shape[0]
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u) - 1.0
        idx = np.arange(1, n + 1)
        cond = u - cssv / idx > 0
        rho_idx = np.nonzero(cond)[0][-1]     # largest index satisfying the condition
        tau = cssv[rho_idx] / (rho_idx + 1)
        return np.maximum(v - tau, 0.0)

    def _inner_descent(self, x_h, mu: np.ndarray, G: Optional[float],
                        penalty_type: Optional[Literal["eq", "ineq"]],
                        lam: float, rho: float, eta: float, n_iter: int):
        """
        Projected gradient descent on x_h (Helmert space) minimizing L_rho with lam and rho
        held fixed -- the inner loop of the augmented-Lagrangian scheme. When penalty_type is
        None the return constraint is skipped entirely and this just descends the GPR surface.
        After every gradient step, x_h is mapped to the simplex, Euclidean-projected onto
        {w >= 0, sum(w) = 1} (see _project_simplex), and mapped back to Helmert space. This
        keeps w feasible for the simplex box at every single step, independently of the return
        constraint, which is instead handled by lam/rho and is only satisfied at convergence.
        Parameters:
        x_h: starting point in Helmert space (n_assets-1,)
        mu, G, penalty_type: return constraint definition (penalty_type=None: unconstrained)
        lam, rho: current multiplier and penalty coefficient (fixed within this inner loop)
        eta: gradient step size.
        n_iter: number of inner gradient steps.
        Returns the final x_h and the per-step surface-value history (for diagnostics).
        """
        x_h = np.asarray(x_h, dtype=float)
        history = []
        for _ in range(n_iter):
            if penalty_type is None:
                grad = self._cost_sign * np.asarray(self.surface_grad(x_h), dtype=float)
            else:
                grad, _ = self._objective_grad(x_h, mu, G, penalty_type, lam, rho)
                grad = np.asarray(grad, dtype=float)
            x_h = x_h - eta * grad
            w = self._project_simplex(np.asarray(self._helmert_to_simplex(x_h), dtype=float))
            x_h = np.asarray(self._simplex_to_helmert(w), dtype=float)
            history.append(float(self.surface(x_h)))
        return x_h, history

    def _augmented_lagrangian_descent(self, x_h0, mu: np.ndarray, G: Optional[float],
                                       penalty_type: Optional[Literal["eq", "ineq"]],
                                       eta: float, rho0: float, rho_growth: float, rho_max: float,
                                       n_inner: int, n_outer: int, tol_feas: float,
                                       verbose: int = 0, restart_label: str = ""):
        """
        Full augmented-Lagrangian scheme (method of multipliers) for a single starting point.
        Two nested loops: the outer loop updates lam between rounds of descent (and moderately
        grows rho, capped at rho_max) so that lam converges to the true Lagrange multiplier; the
        inner loop (_inner_descent) minimizes L_rho with lam and rho held fixed. This is what
        lets the constraint be satisfied with a moderate, "fixed" rho instead of driving a
        single penalty coefficient to infinity.
        When penalty_type is None, the outer loop and the whole lam/rho machinery are skipped:
        a single round of unconstrained projected descent on the surface is run.
        Parameters:
        x_h0: starting point in Helmert space (n_assets-1,)
        mu, G, penalty_type: return constraint definition (penalty_type=None: unconstrained)
        eta: inner gradient step size.
        rho0, rho_growth, rho_max: initial penalty coefficient, per-round growth factor, and cap.
        n_inner, n_outer: number of inner steps per round, and max number of outer rounds.
        tol_feas: stop the outer loop once the constraint violation drops below this tolerance.
        verbose: 0 silent, >=1 prints one summary line at the end of this restart (via
            tqdm.write, so it doesn't fight with the restart-level progress bar), >=2 also
            shows a live per-outer-round progress bar with R, G, the R-G gap, lam, rho and the
            current surface value as postfix. All logged numbers are read off values already
            computed by the descent (R, hist, lam, rho) -- nothing here triggers an extra
            surface/gradient evaluation, so enabling verbose does not change the optimization
            path or its cost.
        restart_label: short prefix (e.g. "[restart 3/10] ") prepended to log lines so multiple
            restarts can be told apart when verbose logging is on.
        Returns the final x_h, the achieved return R = mu . w, and the concatenated per-step
        surface-value history across all outer rounds (for diagnostics).
        """
        x_h = np.asarray(x_h0, dtype=float)
        history = []

        if penalty_type is None:
            x_h, hist = self._inner_descent(x_h, mu, G, None, 0.0, 0.0, eta, n_inner)
            history.extend(hist)
            w = self._helmert_to_simplex(x_h)
            R = float(np.dot(np.asarray(mu, dtype=float), np.asarray(w, dtype=float)))
            if verbose >= 1:
                cost = hist[-1] if hist else float("nan")
                tqdm.write(f"{restart_label}unconstrained | R={R:.4f} | surface={cost:.4f}")
            return x_h, R, history

        lam = 0.0
        rho = rho0
        R = None
        viol = None
        outer_iter = range(n_outer)
        if verbose >= 2:
            outer_iter = tqdm(outer_iter, desc=f"{restart_label}outer rounds", leave=False)
        for _ in outer_iter:
            x_h, hist = self._inner_descent(x_h, mu, G, penalty_type, lam, rho, eta, n_inner)
            history.extend(hist)

            w = self._helmert_to_simplex(x_h)
            R = float(np.dot(np.asarray(mu, dtype=float), np.asarray(w, dtype=float)))
            delta = R - G

            if penalty_type == "eq":
                viol = abs(delta)
            else:  # "ineq": R >= G is only violated when delta < 0
                viol = max(0.0, -delta)

            # multiplier update BETWEEN outer rounds (method of multipliers), not per inner
            # step: this is what drives lam towards the true Lagrange multiplier, so the
            # constraint ends up satisfied without sending rho to infinity
            if penalty_type == "eq":
                lam = lam - rho * delta
            else:
                lam = max(0.0, lam - rho * delta)  # projected to stay non-negative

            if verbose >= 2:
                outer_iter.set_postfix({
                    "R": f"{R:.4f}", "G": f"{G:.4f}", "R-G": f"{delta:+.4f}",
                    "surface": f"{hist[-1]:.4f}", "lam": f"{lam:.3g}", "rho": f"{rho:.3g}",
                    "viol": f"{viol:.2e}",
                })

            if viol < tol_feas:
                break
            rho = min(rho * rho_growth, rho_max)   # moderate growth, capped

        if verbose >= 1:
            delta = R - G
            tqdm.write(
                f"{restart_label}R={R:.4f} target(G)={G:.4f} R-G={delta:+.4f} "
                f"surface={hist[-1]:.4f} lam={lam:.3g} rho={rho:.3g} viol={viol:.2e} "
                f"feasible={viol < tol_feas}"
            )

        return x_h, R, history

    def optimize(self, mu: np.ndarray, G: Optional[float] = None,
                 mode: Optional[Literal["eq", "ineq"]] = None,
                 eta: float = 0.05, rho0: float = 1.0, rho_growth: float = 1.5,
                 rho_max: float = 1e4, n_inner: int = 200, n_outer: int = 30,
                 tol_feas: float = 1e-4, random_state: Optional[int] = None,
                 verbose: int = 0) -> Dict[str, Any]:
        """
        Public entry point: finds the minimum-risk portfolio on the simplex, optionally subject
        to a target return G, running the augmented-Lagrangian scheme from self.n_restarts
        distinct starting points to avoid local minima (the surface is a GPR posterior mean,
        not necessarily convex).
        Parameters:
        mu: expected returns per asset (n_assets,). Always used to report the achieved return,
            even when mode is None.
        G: target return level. Required when mode is "eq" or "ineq"; ignored (may be None)
            when mode is None (no return constraint).
        mode: "eq" for R = G, "ineq" for R >= G, None for no return constraint.
        eta: inner gradient step size.
        rho0, rho_growth, rho_max: augmented-Lagrangian penalty schedule (initial value, growth
            factor per outer round, and cap). Moderate defaults; lam does the heavy lifting so
            these do not need hand-tuning to hit the constraint precisely.
        n_inner, n_outer: inner steps per outer round, and max outer rounds.
        tol_feas: outer-loop stopping tolerance on the constraint violation.
        random_state: seed for the Dirichlet restarts (reproducibility).
        verbose: diagnostic logging level, useful for tuning eta/rho0/rho_growth/n_inner/
            n_outer and for telling apart "didn't converge" from "converged past the target
            return". 0 (default): silent, identical behaviour to before this parameter existed.
            1: a tqdm progress bar over restarts, plus one summary line per restart (achieved R
            vs target G, the R-G gap so overshoot is visible directly, the surface/cost value,
            and feasibility), plus a final line for the selected candidate. 2: additionally, a
            live nested progress bar per restart over outer augmented-Lagrangian rounds, with
            R, G, R-G, surface, lam, rho and violation as postfix, so the walk towards (or past)
            the constraint is visible round by round. All logged values are read off numbers the
            descent already computes for its own bookkeeping (R, history, lam, rho) -- verbose
            logging does not add any extra surface/gradient evaluation, so it never changes the
            optimization path, its cost, or its result.
        Returns a dict with:
            w_star: optimal weights on the simplex (n_assets,)
            x_h_star: corresponding point in Helmert space (n_assets-1,)
            risk_mean: GPR posterior mean (predicted risk) at w_star
            risk_std: GPR posterior standard deviation at w_star (surrogate's confidence)
            R: achieved return mu . w_star
            feasible: whether the return constraint is satisfied within tol_feas
            constraint_violation: the residual violation of the chosen candidate
            mode, G: the constraint settings used
            history: list (one per restart) of surface-value histories, for diagnostics
        """
        if mode is not None and G is None:
            raise ValueError("G must be provided when mode is 'eq' or 'ineq'")
        if mode not in (None, "eq", "ineq"):
            raise ValueError("mode must be None, 'eq' or 'ineq'")

        mu_arr = np.asarray(mu, dtype=float)
        n_assets = self._U.shape[1]
        rng = np.random.default_rng(random_state)

        # multi-restart: n_restarts distinct starting points sampled on the simplex via
        # Dirichlet(1,...,1) (uniform over the simplex, already w >= 0 and sum(w) = 1 by
        # construction, so no projection is needed before mapping to Helmert space)
        w_starts = rng.dirichlet(np.ones(n_assets), size=self.n_restarts)

        candidates = []
        restart_iter = enumerate(w_starts)
        if verbose >= 1:
            restart_iter = tqdm(restart_iter, total=self.n_restarts, desc="EFOptimizer restarts")
        for i, w0 in restart_iter:
            label = f"[restart {i + 1}/{self.n_restarts}] "
            x_h0 = np.asarray(self._simplex_to_helmert(w0), dtype=float)
            x_h_star, _, hist = self._augmented_lagrangian_descent(
                x_h0, mu_arr, G, mode, eta=eta, rho0=rho0, rho_growth=rho_growth,
                rho_max=rho_max, n_inner=n_inner, n_outer=n_outer, tol_feas=tol_feas,
                verbose=verbose, restart_label=label,
            )
            # guard against residual floating-point drift off the simplex before reporting
            w_star = self._project_simplex(np.asarray(self._helmert_to_simplex(x_h_star), dtype=float))
            x_h_star = np.asarray(self._simplex_to_helmert(w_star), dtype=float)
            R_star = float(np.dot(mu_arr, w_star))
            risk_mean = float(self.surface(x_h_star))  # closed-form surface (raw, un-flipped), used only to rank candidates

            if mode == "eq":
                viol = abs(R_star - G)
            elif mode == "ineq":
                viol = max(0.0, G - R_star)
            else:
                viol = 0.0
            feasible = viol < tol_feas

            candidates.append({
                "w": w_star, "x_h": x_h_star, "risk_mean": risk_mean,
                "R": R_star, "violation": viol, "feasible": feasible, "history": hist,
            })

            if verbose >= 1 and hasattr(restart_iter, "set_postfix"):
                # best-so-far among restarts, purely for the live bar -- reuses risk_mean/
                # feasible/viol already computed above, no extra evaluation
                feas_so_far = [c for c in candidates if c["feasible"]]
                pool = feas_so_far if feas_so_far else candidates
                best_so_far = min(pool, key=lambda c: (not c["feasible"], self._cost_sign * c["risk_mean"]))
                restart_iter.set_postfix({
                    "best_R": f"{best_so_far['R']:.4f}", "best_cost": f"{best_so_far['risk_mean']:.4f}",
                    "best_feasible": best_so_far["feasible"],
                })

        # selection: feasibility first; among feasible candidates pick the lowest predicted
        # cost (cost_sign * risk_mean, so this is genuinely the safest candidate regardless of
        # whether higher_is_better means the raw surrogate target should be maximized or
        # minimized); if none is feasible, fall back to the least-violating one
        feasible_candidates = [c for c in candidates if c["feasible"]]
        if feasible_candidates:
            best = min(feasible_candidates, key=lambda c: self._cost_sign * c["risk_mean"])
        else:
            best = min(candidates, key=lambda c: (c["violation"], self._cost_sign * c["risk_mean"]))

        # uncertainty: query the fitted GPR itself (not the closed-form surface, which is
        # mean-only) at w_star for the posterior mean and standard deviation -- the std is a
        # direct measure of how confident the surrogate is about this particular portfolio,
        # typically higher in under-sampled regions of the simplex
        mean_pred, std_pred = self.surrogate.predict(best["w"][None, :], return_std=True)

        if verbose >= 1:
            gap = "n/a" if G is None else f"{best['R'] - G:+.4f}"
            g_str = "n/a" if G is None else f"{G:.4f}"
            tqdm.write(
                f"==> selected: R={best['R']:.4f} target(G)={g_str} R-G={gap} "
                f"risk_mean={float(np.ravel(mean_pred)[0]):.4f} risk_std={float(np.ravel(std_pred)[0]):.4f} "
                f"feasible={best['feasible']} violation={best['violation']:.2e}"
            )

        return {
            "w_star": best["w"],
            "x_h_star": best["x_h"],
            "risk_mean": float(np.ravel(mean_pred)[0]),
            "risk_std": float(np.ravel(std_pred)[0]),
            "R": best["R"],
            "feasible": best["feasible"],
            "constraint_violation": best["violation"],
            "mode": mode,
            "G": G,
            "history": [c["history"] for c in candidates],
        }

