from ctypes import CDLL, POINTER, byref, c_bool, c_double, c_int

import numpy as np

from lib.utils import get_root

# constants
DLL_FOLDER = "lib/clib"
DLL_NAME = "custom_cg.so"

# path to root directory
root = get_root()

# shared library path
lib_path = root / DLL_FOLDER / DLL_NAME

# load shared library
custom_cg_lib = CDLL(lib_path.as_posix())


class CustomCG:

    def __init__(
        self,
        A: np.ndarray,
        b: np.ndarray,
        x_0: np.ndarray,
        tol: float = 1e-6,
        maxiter=100,
    ):
        # system
        self.A = A
        self.b = b

        # initial guess
        self.x_0 = x_0

        # initial residual
        self.r_0 = b - A @ x_0
        self.rho_0 = -1.0  # initial residual in the eigenbasis of A

        # convergence criteria
        self.tol = tol
        self.maxiter = maxiter

        # solution
        self.x_m = np.zeros_like(x_0)

        # intermediate results
        self.x_i = np.zeros((self.maxiter + 1, self.A.shape[0]), dtype=np.float64)

        # exact solution
        self.x_exact = np.zeros_like(x_0)
        self.exact_convergence = False

        # number of iterations required
        self.niters = 0

        # cg coefficients
        self.alpha = np.zeros(maxiter, dtype=np.float64)
        self.beta = np.zeros(maxiter, dtype=np.float64)

        # A eigen spectrum
        self.eigenvalues = np.zeros(A.shape[0], dtype=np.float64)
        self.eigenvectors = np.zeros(A.shape, dtype=np.float64)

        # search directions
        self.search_directions = np.zeros((self.maxiter, A.shape[0]), dtype=np.float64)

        # polynomial coefficients
        self.residual_polynomials_coefficients = []

    def solve(
        self,
        save_iterates: bool = False,
        save_search_directions: bool = False,
        x_exact: np.ndarray = np.array([]),
    ) -> tuple[np.ndarray, bool]:
        # convert to c types
        n = c_int(self.A.shape[0])
        tol = c_double(self.tol)
        maxiter = c_int(self.maxiter)
        A = self.A.flatten().astype(c_double)
        b = self.b.astype(c_double)
        x_0 = self.x_0.astype(c_double)
        alpha = self.alpha.astype(c_double)
        beta = self.beta.astype(c_double)
        niters = c_int(0)
        save_iterates_c = c_bool(save_iterates)
        iterates = self.x_i.flatten().astype(c_double)
        search_directions = self.search_directions.flatten().astype(c_double)
        x_exact = x_exact.astype(c_double)
        if x_exact.size > 0:
            self.x_exact = x_exact
            self.exact_convergence = True
        exact_convergence = c_bool(self.exact_convergence)

        # allocate memory for solution
        x = np.zeros_like(self.x_0).astype(c_double)

        # set function signature
        custom_cg_lib.custom_cg.argtypes = [
            POINTER(c_double),  # A
            POINTER(c_double),  # b
            POINTER(c_double),  # x_0
            POINTER(c_double),  # x_m
            POINTER(c_double),  # alpha
            POINTER(c_double),  # beta
            c_int,  # size
            POINTER(c_int),  # niters
            c_int,  # maxiter
            c_double,  # tol
            c_bool,  # save_iterates
            POINTER(c_double),  # iterates
            c_bool,  # safe_search_directions
            POINTER(c_double),  # P
            c_bool,  # exact_convergence
            POINTER(c_double),  # x_exact
        ]
        custom_cg_lib.custom_cg.restype = c_bool

        # call the function
        success = custom_cg_lib.custom_cg(
            A.ctypes.data_as(POINTER(c_double)),
            b.ctypes.data_as(POINTER(c_double)),
            x_0.ctypes.data_as(POINTER(c_double)),
            x.ctypes.data_as(POINTER(c_double)),
            alpha.ctypes.data_as(POINTER(c_double)),
            beta.ctypes.data_as(POINTER(c_double)),
            n,
            byref(niters),
            maxiter,
            tol,
            save_iterates_c,
            iterates.ctypes.data_as(POINTER(c_double)),
            c_bool(save_search_directions),
            search_directions.ctypes.data_as(POINTER(c_double)),
            exact_convergence,
            x_exact.ctypes.data_as(POINTER(c_double)),
        )

        # save solution
        x = np.ctypeslib.as_array(x)
        self.x_m = x

        # save number of iterations
        self.niters = int(niters.value)

        # save coefficients
        self.alpha = np.ctypeslib.as_array(alpha)[: self.niters]
        self.beta = np.ctypeslib.as_array(beta)[: (self.niters - 1)]

        # save iterates
        if save_iterates:
            self.x_i = np.ctypeslib.as_array(iterates).reshape(
                (self.maxiter + 1, self.A.shape[0])
            )[: self.niters + 1]
            self.x_i[0] = x_0

        # save search directions
        if save_search_directions:
            self.search_directions = np.ctypeslib.as_array(search_directions).reshape(
                (self.maxiter, self.A.shape[0])
            )[: self.niters]

        return x, success

    def cg_polynomial(
        self,
        resolution: int,
        domain: tuple = (),
        eig_domain: tuple = (),
        respoly_error: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # calculate respoly coefficients
        if len(self.residual_polynomials_coefficients) == 0:
            self.residual_polynomials()

        # degree of the polynomial grows with every CG iteration and so does the number of coefficients
        degree = self.niters + 1

        # specify domain and resolution
        no_domain = domain == ()
        no_eig_domain = eig_domain == ()
        if no_eig_domain and no_domain:
            raise ValueError("Specify either eig_domain or domain.")

        if no_domain:
            eigenvalue_range = eig_domain[1] - eig_domain[0]
            domain_min = eig_domain[0] - 0.2 * eigenvalue_range
            domain_max = eig_domain[1] + 0.2 * eigenvalue_range
            domain = (domain_min, domain_max)

        # create x values
        x = np.linspace(domain[0], domain[1], resolution)

        # create residual polynomial & absolute error
        r = np.zeros((degree, resolution))
        for i in range(0, degree):
            coeffs = self.residual_polynomials_coefficients[i]
            rp = np.poly1d(coeffs)
            r[i] = rp(x)

        if respoly_error:
            e = self.calculate_respoly_error()
        else:
            e = np.empty(self.niters + 1)

        return x, r, e

    def residual_polynomials(self) -> list[np.ndarray]:
        delta = 1 / self.alpha + np.append(0, self.beta / self.alpha[:-1])
        eta = np.append(0, np.sqrt(self.beta) / self.alpha[:-1])
        r_polynomials_coefficients = [[1], [1, -delta[0]]]
        rp_previous = np.poly1d(r_polynomials_coefficients[0])
        rp_current = np.poly1d(r_polynomials_coefficients[1])
        for i in range(2, self.niters + 1):
            # construct the next residual polynomial
            p = np.poly1d([1, -delta[i - 1]])
            p = np.polymul(p, rp_current)
            p = np.polyadd(p, -eta[i - 2] * rp_previous)
            p = p / eta[i - 1]
            r_polynomials_coefficients.append(list(p.coeffs))

            # update previous and current residual polynomials
            rp_previous = rp_current
            rp_current = p

        # normalize the coefficients
        r_polynomials_coefficients = [
            np.array(c) / c[-1] for c in r_polynomials_coefficients
        ]

        self.residual_polynomials_coefficients = r_polynomials_coefficients

        return r_polynomials_coefficients

    def calculate_respoly_error(self) -> np.ndarray:
        if self.residual_polynomials_coefficients == []:
            self.residual_polynomials()

        self.eigenvalues, self.eigenvectors = np.linalg.eig(self.A)

        # calculate initial residual in the eigenbasis of A
        self.rho_0 = self.eigenvectors.T @ self.r_0
        e = np.zeros(self.niters + 1)
        for i in range(self.niters + 1):
            coeffs = self.residual_polynomials_coefficients[i]
            rp = np.poly1d(coeffs)
            r_lambda = rp(self.eigenvalues)
            e[i] = np.sqrt(np.sum(r_lambda**2 * self.rho_0**2 / self.eigenvalues))

        return e

    # helper
    def calculate_errors(self, x_exact: np.ndarray) -> np.ndarray:
        if self.x_i is None:
            raise ValueError("No iterates saved")

        # calculate A-norm errors
        eps = x_exact - self.x_i
        return np.sqrt(np.sum(eps * (self.A @ eps.T).T, axis=1))

    def calculate_residuals(self) -> np.ndarray:
        if self.x_i is None:
            raise ValueError("No iterates saved")

        # calculate residuals
        return self.b - (self.A @ self.x_i.T).T

    def calculate_iteration_upperbound(self) -> int:
        """
        Assumes eigenvalues are uniformly distributed between lowest and highest eigenvalue. In this case, the
        classical CG convergence factor is given by f = (sqrt(cond) - 1) / (sqrt(cond) + 1), where cond is the condition
        number of A. The number of iterations required to reach a tolerance tol is given by ceil(log(tol / (2 * e0_Anorm)) / log(f)).
        """
        # convergence factor
        cond = np.linalg.cond(self.A)
        sqrt_cond = np.sqrt(cond)
        convergence_factor = (np.sqrt(cond) - 1) / (np.sqrt(cond) + 1)

        # convergence tolerance
        conv_tol = np.log(self.tol / 2)
        if self.exact_convergence:  # See report Theorem: "Residual convergence criterion"
            conv_tol -= np.log(sqrt_cond)

        return int(np.ceil(conv_tol / np.log(convergence_factor)))

    @staticmethod
    def calculate_iteration_upperbound_static(cond: float, log_rtol: float, exact_convergence: bool = False) -> int:
        """
        Static version of the calculate_iteration_upperbound method.
        """
        # convergence factor
        sqrt_cond = np.sqrt(cond)
        convergence_factor = (sqrt_cond - 1) / (sqrt_cond + 1)

        # convergence tolerance
        conv_tol = log_rtol - np.log(2)
        if exact_convergence: # See report Theorem: "Residual convergence criterion"
            conv_tol -= np.log(sqrt_cond)

        return int(np.ceil(conv_tol / np.log(convergence_factor)))

    def calculate_improved_cg_iteration_upperbound(
        self,
        clusters: list[tuple[float, float]],
    ) -> int:
        """
        Calculates an improved CG iteration bound for non-uniform eigenspectra.
        Assumes available knowledge on the whereabouts of eigenvalue clusters
        """
        # setup
        e0 = self.x_exact - self.x_0
        e0_Anorm = np.sqrt(np.sum(e0 * (self.A @ e0.T).T))
        log_rtol = np.log(self.tol / (2 * e0_Anorm))
        degrees = [0] * len(clusters)

        for i, cluster in enumerate(clusters):
            a_i, b_i = cluster
            log_rtol_eff = log_rtol
            for j in range(i):
                a_j, b_j = clusters[j]
                z_1 = (b_j + a_j - 2 * b_i) / (b_j - a_j)
                z_2 = (b_j + a_j) / (b_j - a_j)
                m_j = degrees[j]
                log_rtol_eff -= m_j * (
                    np.log(abs(z_1 - np.sqrt(z_1**2 - 1)) / (z_2 + np.sqrt(z_2**2 - 1)))
                )

            # calculate & store chebyshev degree
            degrees[i] = CustomCG.calculate_iteration_upperbound_static(
                cond=b_i / a_i, log_rtol=log_rtol_eff
            )
        return sum(degrees)

    # make properties
    @property
    def A(self):
        return self._A

    @A.setter
    def A(self, A):
        self._A = A

    @property
    def b(self):
        return self._b

    @b.setter
    def b(self, b):
        self._b = b

    @property
    def x_0(self):
        return self._x_0

    @x_0.setter
    def x_0(self, x_0):
        self._x_0 = x_0

    @property
    def tol(self):
        return self._tol

    @tol.setter
    def tol(self, tol):
        self._tol = tol

    @property
    def maxiter(self):
        return self._maxiter

    @maxiter.setter
    def maxiter(self, maxiter):
        self._maxiter = maxiter


if __name__ == "__main__":
    # test library access
    custom_cg_lib.TEST()

    # test basic arithmetic + function interface with ctypes
    cadd = custom_cg_lib.add
    cadd.argtypes = [POINTER(c_double), POINTER(c_double), POINTER(c_double), c_int]
    # cadd.restype = POINTER(c_double)
    a = np.array([1.1, 2.2, 3.3])
    b = np.array([4.4, 5.5, 6.6])
    c = np.array([0.0, 0.0, 0.0])
    cadd(
        a.ctypes.data_as(POINTER(c_double)),
        b.ctypes.data_as(POINTER(c_double)),
        c.ctypes.data_as(POINTER(c_double)),
        len(a),
    )
    c_array = np.ctypeslib.as_array(c, (len(a),))
    print(c_array)
