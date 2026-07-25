import numpy as np
from typing import Any, Dict, Literal, Optional, Tuple, Type, Union
from scipy.stats import qmc


class LHS:
    """
    Latin Hypercube Sampling over the unit hypercube [0,1]^dim.

    Divides each dimension into n_samples equal-probability bins and places
    exactly one point per bin (one sample per row/column of the grid),
    guaranteeing good one-dimensional coverage regardless of how many
    dimensions are sampled.

    With criterion="maximin", n_candidates independent plans are drawn and
    the one maximizing the smallest pairwise distance between points is
    kept, avoiding the clustering (e.g. near a diagonal) that a single
    random permutation can produce. With criterion="random", the first
    (only) plan drawn is returned as-is.

    Parameters:
    dim (int): number of dimensions of the unit hypercube to sample.
    criterion (Literal["random", "maximin"]): plan-selection rule.
    n_candidates (int): number of candidate plans drawn when
        criterion="maximin"; ignored when criterion="random".
    seed (int, optional): fixes the permutations and offsets drawn.
    """

    def __init__(self,
                 dim: int,
                 criterion: Literal["random", "maximin"] = "maximin",
                 n_candidates: int = 50,
                 seed: Optional[int] = None):
        assert dim > 0, f"dim must be > 0 (current is {dim})"
        assert n_candidates > 0, f"n_candidates must be > 0 (current is {n_candidates})"
        self.dim = dim
        self.criterion = criterion
        self.n_candidates = n_candidates
        self.seed = seed

    def __repr__(self):
        return f"<Object: Callable LHS> (dim={self.dim}; criterion={self.criterion})"

    def _plan(self, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        """Draw a single LHS plan: one random permutation of bins per dimension, offset by a uniform draw within each bin."""
        bins = np.stack([rng.permutation(n_samples) for _ in range(self.dim)], axis=1)
        offsets = rng.uniform(size=(n_samples, self.dim))
        return (bins + offsets) / n_samples

    @staticmethod
    def _min_pairwise_distance(points: np.ndarray) -> float:
        diffs = points[:, None, :] - points[None, :, :]
        dist = np.sqrt((diffs ** 2).sum(axis=-1))
        np.fill_diagonal(dist, np.inf)
        return dist.min()

    def __call__(self, n_samples: int) -> np.ndarray:
        """Returns an (n_samples, dim) array of points in [0,1]^dim."""
        assert n_samples > 0, f"n_samples must be > 0 (current is {n_samples})"
        rng = np.random.default_rng(self.seed)
        if self.criterion == "random" or n_samples < 2:
            return self._plan(n_samples, rng)
        best_plan, best_score = None, -np.inf
        for _ in range(self.n_candidates):
            plan = self._plan(n_samples, rng)
            score = self._min_pairwise_distance(plan)
            if score > best_score:
                best_plan, best_score = plan, score
        return best_plan


class Sobol:
    """
    Sobol low-discrepancy sequence sampling over the unit hypercube [0,1]^dim.

    Points are generated deterministically by a base-2 digital construction
    that minimizes discrepancy (how far the sample departs from a perfectly
    uniform fill of the domain), remaining uniform in higher-dimensional
    projections rather than just along each individual axis. This tends to
    make Sobol the better choice as the number of dimensions grows.

    Parameters:
    dim (int): number of dimensions of the unit hypercube to sample.
    scramble (bool): applies Owen scrambling, which avoids repeating the
        exact same deterministic sequence across independent samplers.
    seed (int, optional): fixes the scrambling; ignored when scramble=False.
    """

    def __init__(self,
                 dim: int,
                 scramble: bool = True,
                 seed: Optional[int] = None):
        assert dim > 0, f"dim must be > 0 (current is {dim})"
        self.dim = dim
        self.scramble = scramble
        self.seed = seed

    def __repr__(self):
        return f"<Object: Callable Sobol> (dim={self.dim}; scramble={self.scramble})"

    def __call__(self, n_samples: int) -> np.ndarray:
        """Returns an (n_samples, dim) array of points in [0,1]^dim."""
        assert n_samples > 0, f"n_samples must be > 0 (current is {n_samples})"
        sampler = qmc.Sobol(d=self.dim, scramble=self.scramble, seed=self.seed)
        return sampler.random(n_samples)


class Map2Simplex:
    """
    Maps points from the unit cube [0,1]^(dim-1) onto the simplex
    Delta^(dim-1) = {x in R^dim : x_i >= 0, sum(x_i) = 1}.

    Each input point's dim-1 coordinates are treated as ordered cuts along
    the unit interval: sorting them and taking consecutive differences
    (including the 0 and 1 endpoints) yields dim non-negative lengths that
    already sum to one. This preserves the coverage properties inherited
    from whichever sampler produced the cuts (LHS or Sobol); dividing raw
    hypercube points by their own sum instead would concentrate samples
    near the simplex's centroid and under-represent its edges and vertices.

    Parameters:
    dim (int): dimension of the simplex's ambient space (i.e. the length
        of each output point); input points must have dim-1 columns.
    """

    def __init__(self, dim: int):
        assert dim > 1, f"dim must be > 1 to define a simplex (current is {dim})"
        self.dim = dim

    def __repr__(self):
        return f"<Object: Callable Map2Simplex> (dim={self.dim})"

    def __call__(self, cuts: np.ndarray) -> np.ndarray:
        """
        cuts (np.ndarray): (n_samples, dim-1) array of points in [0,1]^(dim-1) </p>
        Returns an (n_samples, dim) array of points on the simplex.
        """
        cuts = np.atleast_2d(cuts)
        assert cuts.shape[1] == self.dim - 1, \
            f"cuts must have {self.dim - 1} columns (current is {cuts.shape[1]})"
        n_samples = cuts.shape[0]
        padded = np.empty((n_samples, self.dim + 1))
        padded[:, 0] = 0.0
        padded[:, 1:-1] = np.sort(cuts, axis=1)
        padded[:, -1] = 1.0
        return np.diff(padded, axis=1)


SamplerType = Type[Union[LHS, Sobol]]
MapperType = Type[Map2Simplex]


class SamplerShell:
    """
    Orchestrates a base sampler (LHS or Sobol) together with an optional
    geometry mapper (e.g. Map2Simplex), producing a design of experiments
    over `dim` real dimensions.

    `dim` is always the true dimensionality of the space being sampled.
    When mapper_cls is given, the underlying sampler is built with dim-1
    dimensions instead of dim, since the mapper reconstructs the degree of
    freedom removed by its geometric constraint (e.g. the simplex
    constraint sum(x) = 1 removes one free coordinate). When mapper_cls is
    omitted, the sampler is built directly with dim dimensions and its
    [0,1]^dim output is rescaled to `bounds`.

    Parameters:
    sampler_cls (type): LHS or Sobol, the base [0,1]^d sampler to use.
    dim (int): true number of dimensions of the sampled space.
    mapper_cls (type, optional): geometry mapper applied to the sampler's
        raw output, e.g. Map2Simplex.
    bounds (tuple of (float, float), optional): per-dimension (low, high)
        bounds used to rescale the sampler's [0,1]^dim output; only valid
        when mapper_cls is omitted, since a mapper's geometry already
        fixes the admissible region.
    sampler_kwargs (dict, optional): extra keyword arguments forwarded to
        sampler_cls's constructor (e.g. seed, criterion, scramble).
    """

    def __init__(self,
                 sampler_cls: SamplerType,
                 dim: int,
                 mapper_cls: Optional[MapperType] = None,
                 bounds: Optional[Tuple[Tuple[float, float], ...]] = None,
                 sampler_kwargs: Optional[Dict[str, Any]] = None):
        assert dim > (1 if mapper_cls is not None else 0), \
            f"dim must be > {1 if mapper_cls is not None else 0} for the given mapper_cls (current is {dim})"
        if mapper_cls is not None:
            assert bounds is None, "bounds is ignored when mapper_cls is given; the mapper's geometry fixes the admissible region"
        if bounds is not None:
            assert len(bounds) == dim, f"bounds must have {dim} entries (current is {len(bounds)})"

        self.sampler_cls = sampler_cls
        self.dim = dim
        self.mapper_cls = mapper_cls
        self.bounds = bounds
        self.sampler_kwargs = sampler_kwargs or {}

        sampler_dim = dim - 1 if mapper_cls is not None else dim
        self.sampler = sampler_cls(dim=sampler_dim, **self.sampler_kwargs)
        self.mapper = mapper_cls(dim=dim) if mapper_cls is not None else None

        if bounds is not None:
            self._low = np.array([lo for lo, _ in bounds])
            self._high = np.array([hi for _, hi in bounds])

    def __repr__(self):
        mapper_name = self.mapper_cls.__name__ if self.mapper_cls is not None else None
        return f"<Object: Callable SamplerShell> (dim={self.dim}; sampler={self.sampler_cls.__name__}; mapper={mapper_name})"

    def __call__(self, n_samples: int) -> np.ndarray:
        """Returns an (n_samples, dim) design of experiments."""
        points = self.sampler(n_samples)
        if self.mapper is not None:
            return self.mapper(points)
        if self.bounds is not None:
            return self._low + points * (self._high - self._low)
        return points
