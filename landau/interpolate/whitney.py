"""
RBF interpolator with Whitney-style gradient extension for exterior points.

Interior/exterior detection: half-space test using hull.equations.
Boundary projection: QP (SLSQP) exact projection onto convex hull surface.
Gradient estimation: central finite differences on the RBF interpolant.

Generalizes to N dimensions throughout.

See design and discussion here: https://claude.ai/share/c9d9ca3a-9a9f-48b3-8f2a-886b11951490
"""

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from .basic import TemperatureInterpolator, Interpolation, _CallableInterpolation


def _in_hull(points, hull):
    """
    Boolean interior test using the half-space representation.

    hull.equations rows are [normal | offset] with convention:
        normal · x + offset <= 0  iff  x is inside (or on) the hull.

    Parameters
    ----------
    points : (M, D) ndarray
    hull   : ConvexHull

    Returns
    -------
    inside : (M,) bool
    """
    A = hull.equations[:, :-1]   # (F, D)
    b = hull.equations[:, -1]    # (F,)
    # slack[i, j] > 0 means point i violates facet j
    slack = points @ A.T + b     # (M, F)
    return (slack <= 0).all(axis=1)


def _in_interval(points, x_min, x_max):
    """Interior test for 1D: points inside [x_min, x_max]. (M, 1) -> (M,) bool."""
    x = points[:, 0]
    return (x >= x_min) & (x <= x_max)


def _project_to_hull(x, hull):
    """
    Find the nearest point on the convex hull boundary to exterior point x.

    Solves the convex QP:
        minimise   ||y - x||²
        subject to A·y + b <= 0    (half-space constraints)

    The solution lies on the hull surface because x is exterior (strictly
    violates at least one constraint), so the optimum is on the boundary.

    Parameters
    ----------
    x    : (D,) ndarray — a single exterior point
    hull : ConvexHull

    Returns
    -------
    x_b : (D,) ndarray — nearest point on hull surface
    """
    A = hull.equations[:, :-1]
    b = hull.equations[:, -1]

    res = minimize(
        fun=lambda y: np.sum((y - x) ** 2),
        jac=lambda y: 2 * (y - x),
        x0=x,                          # warm-start from x itself
        method="SLSQP",
        constraints={"type": "ineq", "fun": lambda y: -(A @ y + b),
                                     "jac": lambda y: -A},
    )
    return res.x


def _project_to_interval(x, x_min, x_max):
    """Nearest boundary point for 1D: clip to [x_min, x_max]. (D,) -> (D,)."""
    return np.clip(x, x_min, x_max)


def _rbf_gradient(rbf, x, eps):
    """
    Central finite-difference gradient of rbf at a single point x.

    Parameters
    ----------
    rbf : RBFInterpolator
    x   : (D,) ndarray
    eps : float — step size

    Returns
    -------
    grad : (D,) ndarray
    """
    D = x.shape[0]
    grad = np.empty(D)
    for d in range(D):
        step = np.zeros(D)
        step[d] = eps
        grad[d] = (rbf((x + step)[None])[0] - rbf((x - step)[None])[0]) / (2 * eps)
    return grad


class WhitneyRBFInterpolator(BaseEstimator, RegressorMixin):
    """
    RBF interpolator with Whitney-style C1 gradient extension outside the
    convex hull of the training data.

    Interior points (inside convex hull) are handled by RBFInterpolator.
    Exterior points receive a local linear extension:

        f_ext(x) = f(x_b) + grad_f(x_b) . (x - x_b)

    where x_b is the exact nearest point on the hull surface, found by a
    convex QP, and f(x_b), grad_f(x_b) are evaluated directly on the RBF.

    This guarantees:
    - C0 continuity at the hull boundary (f_ext = f on the hull surface)
    - Locally consistent linear extrapolation along outward-normal rays
    - Smooth gradient transitions along the boundary (inherited from RBF)

    Parameters
    ----------
    kernel : str, default='thin_plate_spline'
        RBF kernel passed to RBFInterpolator.
    smoothing : float, default=0.0
        Smoothing factor. 0 = exact interpolation.
    degree : int or None, default=2
        Polynomial tail degree. None uses the kernel's minimum. Default is 2, which helps smooth gradients
    epsilon : float, default=1.0
        Shape parameter for kernels that require it.
    grad_eps : float, default=1e-4
        Finite-difference step for gradient estimation at boundary points.

    Attributes
    ----------
    rbf_ : RBFInterpolator
        Fitted interior interpolant.
    hull_ : ConvexHull or None
        Convex hull of training data (ND, D>=2). None in the 1D case.
    x_min_, x_max_ : float or None
        Data interval endpoints (1D only).
    n_features_in_ : int
        Dimensionality D of the input space.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.uniform(-2, 2, (300, 2))
    >>> y = np.sin(X[:, 0]) * np.cos(X[:, 1])
    >>> model = WhitneyRBFInterpolator().fit(X, y)
    >>> model.predict(np.array([[3.0, 0.0], [0.0, 0.0]]))
    """

    def __init__(
        self,
        kernel="thin_plate_spline",
        smoothing=0.0,
        degree=2,
        epsilon=1.0,
        grad_eps=1e-4,
    ):
        self.kernel = kernel
        self.smoothing = smoothing
        self.degree = degree
        self.epsilon = epsilon
        self.grad_eps = grad_eps

    def fit(self, X, y):
        """
        Fit the interpolator.

        Parameters
        ----------
        X : (N, D) array-like
        y : (N,)  array-like

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D, got shape {y.shape}")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of rows")

        self.n_features_in_ = X.shape[1]

        rbf_kwargs = dict(kernel=self.kernel, smoothing=self.smoothing,
                          epsilon=self.epsilon)
        if self.degree is not None:
            rbf_kwargs["degree"] = self.degree

        self.rbf_ = RBFInterpolator(X, y, **rbf_kwargs)

        if X.shape[1] == 1:
            self.hull_  = None
            self.x_min_ = float(X.min())
            self.x_max_ = float(X.max())
        else:
            self.hull_  = ConvexHull(X)
            self.x_min_ = None
            self.x_max_ = None

        return self

    def predict(self, X):
        """
        Predict at query points.

        Parameters
        ----------
        X : (M, D) array-like

        Returns
        -------
        z : (M,) ndarray
        """
        check_is_fitted(self)
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}"
            )

        inside = (
            _in_interval(X, self.x_min_, self.x_max_)
            if self.n_features_in_ == 1
            else _in_hull(X, self.hull_)
        )
        z = np.empty(len(X))

        # --- Interior: direct RBF evaluation (batched) ---
        if inside.any():
            z[inside] = self.rbf_(X[inside])

        # --- Exterior: Whitney linear extension (per point) ---
        for i in np.where(~inside)[0]:
            x   = X[i]
            x_b = (
                _project_to_interval(x, self.x_min_, self.x_max_)
                if self.n_features_in_ == 1
                else _project_to_hull(x, self.hull_)
            )
            f_b  = self.rbf_(x_b[None])[0]
            g_b  = _rbf_gradient(self.rbf_, x_b, self.grad_eps)
            z[i] = f_b + g_b @ (x - x_b)

        return z

    def score(self, X, y):
        """R² score."""
        from sklearn.metrics import r2_score
        return r2_score(y, self.predict(X))


@dataclass(frozen=True, eq=True)
class WhitneyTemperatureInterpolator(TemperatureInterpolator):
    """
    A :class:`TemperatureInterpolator` backed by :class:`WhitneyRBFInterpolator`.

    Fits a 1-D Whitney RBF on temperature data and returns a callable that
    extrapolates smoothly outside the training range.

    .. warning::

        Fitting this interpolator can be very slow, especially when taking data e.g. from calphy as is.
        It's better to pre-smooth raw free energies with a plain interpolator or sub set data and then feeding it to
        this class.

    Parameters
    ----------
    kernel : str, default='thin_plate_spline'
    smoothing : float, default=0.0
    degree : int, default=2
    epsilon : float, default=1.0
    grad_eps : float, default=1e-4
    """

    kernel: str = "thin_plate_spline"
    smoothing: float = 0.0
    degree: int = 2
    epsilon: float = 1.0
    grad_eps: float = 1e-4

    def fit(self, T, y):
        """
        Fit the interpolator.

        Parameters
        ----------
        T : (N,) array-like — temperature values
        y : (N,) array-like — target values

        Returns
        -------
        callable : (M,) ndarray -> (M,) ndarray
        """
        T = np.asarray(T, dtype=float).reshape(-1, 1)
        y = np.asarray(y, dtype=float)
        interp = WhitneyRBFInterpolator(
            kernel=self.kernel,
            smoothing=self.smoothing,
            degree=self.degree,
            epsilon=self.epsilon,
            grad_eps=self.grad_eps,
        ).fit(T, y)

        def predict(t):
            t = np.atleast_1d(np.asarray(t, dtype=float)).reshape(-1, 1)
            return interp.predict(t)

        return _CallableInterpolation(predict)
