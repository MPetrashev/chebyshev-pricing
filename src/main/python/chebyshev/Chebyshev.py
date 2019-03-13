from typing import List

from numpy.polynomial.polyutils import mapdomain
import numpy.polynomial.chebyshev as cheb
import numpy as np
from inspect import signature


class Chebyshev:

    def __init__(self, polynomials, degree, domain, chebpts) -> None:
        self.polynomials = polynomials
        self.degree = degree
        self.domain = domain
        self.chebpts = chebpts
        self.levels = mapdomain(chebpts, cheb.Chebyshev.window, domain[-1])  # review for >3d dimension

    @staticmethod
    def _chebinterpolate(chebpts, values):
        order = len(chebpts)
        m = cheb.chebvander(chebpts, order-1)
        c = np.dot(m.T, values)
        c[0] /= order
        c[1:] /= 0.5 * order
        return c

    @classmethod
    def interpolate(cls, func, degree, domain=None, args: List=[]):
        num_of_dims = len(signature(func).parameters) - len(args)
        chebpts = cheb.chebpts1(degree + 1)
        if num_of_dims == 1:
            if not domain:
                domain = cheb.Chebyshev.window
            # Use just standard implementation
            values = np.vectorize(lambda x: func(x, *args))(mapdomain(chebpts, cheb.Chebyshev.window, domain))
            coef = Chebyshev._chebinterpolate(chebpts, values)
            return cheb.Chebyshev(coef, domain=domain)
        else:
            func = np.vectorize(func)
            if not domain:
                domain = [cheb.Chebyshev.window] * num_of_dims

            def create_polynomials(level, params):
                levels = mapdomain(chebpts, cheb.Chebyshev.window, domain[level])
                if level == 1:
                    return [cheb.Chebyshev.interpolate(func, degree, domain[0], [y] + params) for y in levels]
                else:
                    return [create_polynomials(level-1, [y] + params) for y in levels]

            polynomials = create_polynomials(num_of_dims-1, args)
            return cls(polynomials, degree, domain, chebpts)

    @staticmethod
    def _get_sub_grid(x, polynomials):
        return [Chebyshev._get_sub_grid(x, sub_polynomials) for sub_polynomials in polynomials] \
            if isinstance(polynomials[0], list) else [p(x) for p in polynomials]

    @staticmethod
    def _interpolate_and_value(chebpts, values, x, domain):
        coef = Chebyshev._chebinterpolate(chebpts, values)
        f = cheb.Chebyshev(coef, domain=domain)
        return f(x)

    def _slice_values(self, values, offset, *args):
        if offset > 1:
            values = [self._slice_values(v, offset - 1, *args) for v in values]
        return Chebyshev._interpolate_and_value(self.chebpts, values, args[offset], self.domain[offset])

    def __call__(self, *args):
        values = Chebyshev._get_sub_grid(args[0], self.polynomials)
        return self._slice_values(values, len(args)-1, *args)
