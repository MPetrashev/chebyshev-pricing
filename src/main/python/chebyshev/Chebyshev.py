from typing import List

from numpy.polynomial.polyutils import mapdomain
import numpy.polynomial.chebyshev as cheb
from bisect import bisect_left
import numpy as np
from inspect import signature
from functools import partial
import collections.abc


class Chebyshev:

    def __init__(self, polynomials, degree, domain,chebpts) -> None:
        self.polynomials = polynomials
        self.degree = degree
        self.domain = domain
        self.chebpts = chebpts
        self.levels = mapdomain(chebpts, cheb.Chebyshev.window, domain[-1])#review for >3d dimension

    def _get_polynomial_cube(self, level):
        ### Returns the cube of polynomials for all sub-levels starting from level `level`
        ###
        return 0

    @staticmethod
    def _chebinterpolate(chebpts,values):
        order = len(chebpts)
        m = cheb.chebvander(chebpts, order-1)
        c = np.dot(m.T, values)
        c[0] /= order
        c[1:] /= 0.5 * order

        return c

    @staticmethod
    def _interpolate_and_value(chebpts, values, x, domain):
        coef = Chebyshev._chebinterpolate(chebpts, values)
        f = cheb.Chebyshev(coef, domain=domain)
        return f(x)

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
                domain = [cheb.Chebyshev.window] * (num_of_dims)

            def create_polynomials(level, args):
                levels = mapdomain(chebpts, cheb.Chebyshev.window, domain[level])
                if level == 1:
                    retVal = [cheb.Chebyshev.interpolate(func, degree, domain[0], [y] + args) for y in levels]
                    return retVal
                else:
                    return [create_polynomials(level-1, [y] + args) for y in levels]

            polynomials = create_polynomials(num_of_dims-1, args)
            return cls(polynomials, degree, domain,chebpts)

    @staticmethod
    def _get_sub_grid(x,polynomials):
        if isinstance(polynomials[0],list):
            return [Chebyshev._get_sub_grid(x,sub_polynomials) for sub_polynomials in polynomials]
        else:
            return [p(x) for p in polynomials]

    def __call__(self, *args):
        x = args[0]
        y = args[1]

        values = Chebyshev._get_sub_grid(x, self.polynomials)
        if len(args) == 2:
            return Chebyshev._interpolate_and_value(self.chebpts, values, y, self.domain[1])
        else:
            # calculate values for interpolation on N-1 cubes
            values2 = [Chebyshev._interpolate_and_value(self.chebpts, v, y, self.domain[1]) for v in values]
            return Chebyshev._interpolate_and_value(self.chebpts, values2, args[2], self.domain[2])
