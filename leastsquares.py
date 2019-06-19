"""Implementation of several Least-Squares algorithms."""
import numpy as np
import numdifftools as nd

from scipy import optimize


class LeastSquaresBase:
    """
    Fit data `x`, `y` using Least Squares algorithm.

    Attributes
    ----------
    x : ndarray
        Points at which data are observed.
    y : ndarray
        Observed data. Notice that x and y must have the same length.
    noise : float
        Estimate of the noise level. Default is 1.0.
    """
    def __init__(self, x, y, noise=1.0):
        self.x = x
        self.y = y
        self.noise = noise

        assert len(x) == len(y)

        self.theta = None
        self._residuals = None
        self.errors = None
        self.noise_est = None

        self.name = 'LeastSquaresBase'

    @property
    def result(self):
        """Compute ideal noises results."""
        return self.compute_ideal()

    @property
    def residuals(self):
        """Return residuals."""
        if self.theta is None:
            raise Exception('Run `fit` first!')
        else:
            return self._residuals

    def fit(self):
        """Estimate parameters of the model."""
        theta0 = [1, 1]
        optimres = optimize.minimize(self._obj, theta0)

        self.theta = optimres.x

        # Estimate noise
        resid = self.y - self.compute_ideal()
        self.noise_est = resid.std(ddof=2)
        self._residuals = resid

        inv = np.linalg.inv

        noise = self.noise
        #self.noise = self.noise_est
        hess = nd.Hessian(self._obj)
        hess_eval = hess(self.theta)
        delta = 1e-12
        hess_eval[0, 0] += delta
        hess_eval[1, 1] += delta
        cov_mat = inv(hess_eval)
        self.noise = noise

        assert cov_mat.shape == (2, 2)

        sqrt = np.sqrt
        self.errors = 1.96*sqrt(cov_mat[0, 0]), 1.96*sqrt(cov_mat[1, 1])

    def _obj(self, theta):
        """Compute negative of log-likelihood function."""
        raise Exception('Not implemented')

    def compute_ideal(self, x=None):
        """Compute ideal observations based on parameter estimation."""
        if x is None:
            x = self.x

        return self.theta[0] * x + self.theta[1]

    def print_result(self):
        """Print found results."""
        m, b = self.theta
        m_err, b_err = self.errors
        print('{}: m = {:.1f}±{:.1f}, b = {:.1f}±{:.1f}, noise_est = {:.1f}'.format(
            self.name, m, m_err, b, b_err, self.noise_est))


class OLS(LeastSquaresBase):
    """Fit data `x`, `y` using Ordinary Least Squares (OLS)."""

    def __init__(self, x, y, noise=1.0):
        super().__init__(x, y, noise)
        self.name = 'LS, ordinary'

    def _obj(self, theta):
        f = theta[0] * self.x + theta[1]

        sigma = self.noise
        resid = (f - self.y) / sigma

        # L2-norm of the residual.
        result = 0.5*np.sum(resid**2)

        return result

    def posterior_pdf(self):
        resid = self.residuals
        resid = np.linspace(resid.min(), resid.max(), num=51)
        sigma = self.noise
        pi, sqrt, exp, prod = np.pi, np.sqrt, np.exp, np.prod

        N = len(resid)
        coeff = (sigma * sqrt(2*pi))**N
        coeff = 1.0 / coeff
        exponents = exp(-resid**2 / (2.0*sigma**2))
        pdf = coeff * prod(exponents)
        return pdf


class LSConservativeFormulation(LeastSquaresBase):
    """
    Fit data `x`, `y` using Least Squares with conservative formulation.
    """

    def __init__(self, x, y, noise=1.0):
        super(__class__, self).__init__(x, y, noise)
        self.name = 'LS, extension 1'

    def _obj(self, theta):
        f = theta[0] * self.x + theta[1]
        sigma0 = self.noise
        resid = (f - self.y) / sigma0

        resid2 = resid**2
        frac = (1 - np.exp(-resid2/2.0)) / resid2
        result = -np.sum(np.log(frac))

        return result

    def posterior_pdf(self):
        resid = self.residuals
        sigma = self.noise
        pi, sqrt, exp, prod = np.pi, np.sqrt, np.exp, np.prod

        N = len(resid)
        coeff = (sigma * sqrt(2*pi))**N
        coeff = 1.0 / coeff
        resid = resid/sigma
        resid2 = resid**2
        exponents = (1 - exp(-resid2 / 2.0)) / resid2
        pdf = coeff * prod(exponents)

        return pdf / pdf.max()


class BadAndGoodLS(LeastSquaresBase):
    def __init__(self, x, y, noise=1.0, beta=0.01, gamma=50):
        super(__class__, self).__init__(x, y, noise)

        self.beta = beta
        self.gamma = gamma

        name = r'LS, extension 2, $\beta = {:.2f}$, $\gamma = {:.2f}$'
        self.name = name.format(self.beta, self.gamma)
        self.short_name = r'LS, extension 2'

    def _obj(self, theta):
        f = theta[0] * self.x + theta[1]
        sigma0 = self.noise
        resid = (f - self.y) / sigma0

        resid2 = resid**2

        beta = self.beta
        gamma = self.gamma

        term_1 = (beta / gamma) * np.exp(-resid2/2.0/gamma**2)
        term_2 = (1 - beta) * np.exp(-resid2/2.0)

        result = -np.sum(np.log(term_1 + term_2))

        return result
