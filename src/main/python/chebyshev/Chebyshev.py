from typing import List

from numpy.polynomial.polyutils import mapdomain
import numpy.polynomial.chebyshev as cheb
from bisect import bisect_left
import numpy as np


class Chebyshev:

    def __init__(self, func, degree, x_domain, y_domain, y_levels, args) -> None:
        self.func = func
        self.degree = degree
        self.y_domain = y_domain
        self.y_levels = y_levels
        self.interpolants = [cheb.Chebyshev.interpolate(func, degree, x_domain, [y] + args) for y in y_levels]
        self.args = args

    @classmethod
    def interpolate(cls, func, degree, domains=None, args: List=[]):
        xcheb = cheb.chebpts1(degree+1)
        y_levels = mapdomain(xcheb, cheb.Chebyshev.window, domains[1])
        return cls(func, degree, domains[0], domains[1], y_levels, args)

    def __call__(self, *args):
        x = args[0]

        def f(y):
            idx = bisect_left(self.y_levels, y)
            return self.interpolants[idx](x)

        obj = cheb.Chebyshev.interpolate(np.vectorize(f), self.degree, self.y_domain)
        return obj(args[1])
