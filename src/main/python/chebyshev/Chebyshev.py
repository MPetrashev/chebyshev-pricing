from typing import List

from numpy.polynomial.polyutils import mapdomain
import numpy.polynomial.chebyshev as cheb
from bisect import bisect_left
import numpy as np
from inspect import signature
import collections.abc


class Chebyshev:

    def __init__(self, polynomials, degree, domain) -> None:
        self.polynomials = polynomials
        self.degree = degree
        self.domain = domain
        self.levels = mapdomain(cheb.chebpts1(degree + 1), cheb.Chebyshev.window, domain[-1])#review for >3d dimension

    @classmethod
    def interpolate(cls, func, degree, domain=None, args: List=[]):
        num_of_dims = len(signature(func).parameters) - len(args)
        func = np.vectorize(func)
        if num_of_dims == 1:
            # Use just standard implementation
            return cheb.Chebyshev.interpolate(func, degree, domain=domain, args=args)
        else:
            if not domain:
                domain = [cheb.Chebyshev.window] * (num_of_dims-1)

            y_levels = mapdomain(cheb.chebpts1(degree + 1), cheb.Chebyshev.window, domain[-1])
            # for y in y_levels:
            #     p = cheb.Chebyshev.interpolate(func, degree, domain[0], [y] + args)
            #     print('x=0, y={}, p={}'.format(y,p(0)))
            polynomials = [cheb.Chebyshev.interpolate(func, degree, domain[0], [y] + args) for y in y_levels]
            return cls(polynomials, degree, domain)

            # # Otherwise use recursive implementation
            # xcheb = cheb.chebpts1(degree + 1)
            # if not domain:
            #     y_domain = cheb.Chebyshev.window
            #     domain = [y_domain] * num_of_dims
            #     y_levels = xcheb
            # else:
            #     if not isinstance(domain[0], collections.abc.Sequence):
            #         domain = [domain]
            #     y_domain = domain[1]
            #     y_levels = mapdomain(xcheb, cheb.Chebyshev.window, domain[1])
            #
            # return cls(np.vectorize(func), degree, domain[0], y_domain, y_levels, args)

    def __call__(self, *args):
        x = args[0]
        points = cheb.chebpts1(self.degree + 1)
        def f(y):
            idx = bisect_left(self.levels, y)
            return self.polynomials[idx](x)

        obj = cheb.Chebyshev.interpolate(np.vectorize(f), self.degree, self.domain[-1])
        return obj(args[-1])
