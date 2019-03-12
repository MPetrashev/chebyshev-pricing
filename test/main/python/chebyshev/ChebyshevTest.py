import math
import unittest
import numpy.polynomial.chebyshev as cheb
from chebyshev.Chebyshev import Chebyshev
import numpy as np
# from rpy2.robjects import r
from itertools import product

from chebyshev.blackscholes import call_price


class ChebyshevTest(unittest.TestCase):

    # @classmethod
    # def setUpClass(cls) -> None:
    #     import rpy2.robjects.packages as rpackages
    #     utils = rpackages.importr('utils')
    #     if not rpackages.isinstalled('chebpol'):
    #         utils.install_packages('chebpol', repos='http://cran.us.r-project.org')

    def testPolynom(self):
        # # check Chebyshev points
        # r_cheb_pts = list(r('library(chebpol)\n' +
        #                   'chebknots(11,intervals=c(-1,1))')[0])
        # r_cheb_pts.reverse()
        # cheb_pts = cheb.chebpts1(11)
        # np.testing.assert_almost_equal(r_cheb_pts,cheb_pts)

        #check function values on Chebyshev points
        def f(x):
            return x*x + 2

        obj = cheb.Chebyshev.interpolate(np.vectorize(f), 10, domain=[0, 100])
        self.assertAlmostEqual(27, obj(5), delta=0.001)
        # r_values = list(r('f <- function(x) x*x + 2\n'
        #                   + 'evalongrid(f,11,intervals=c(-1,1))') )
        # np.testing.assert_almost_equal(r_values,obj(cheb_pts))

        #check Chebyshev coefficients
        # r_coefs = r('chebcoef( evalongrid(f,11,intervals=c(-1,1)) )')
        # obj = cheb.Chebyshev(r_coefs)
        # np.testing.assert_almost_equal(r_values,obj(cheb_pts))

    def test2DPolynom(self):
        def g(x,y):
            return x + y * 10
        # r_value = list(r("library(chebpol)\n"
        #               + "g <- function(x) x[1]  + x[2]*10\n"
        #               + "ch <- ipol(g,dims=c(5,5),method='cheb')\n"
        #               + "ch(c(2.,1.))") )
        # self.assertAlmostEqual(g(2.,1.),r_value[0])
        obj = cheb.Chebyshev.interpolate(np.vectorize(g), 10, domain=[[0, 100],[0, 10]])
        self.assertAlmostEqual(g(2.,1.),obj(2.,1.))

    # def testNDimension(self):
    #     r_coefs = np.asarray(r('library(chebpol)\n'
    #                 + 'g <- function(x) x[1] + x[2]^2 + exp(x[3]) + cos(x[4])\n'
    #                 + 'chebcoef( evalongrid(g,c(5,5,5,5)) )'))
    #     obj = cheb.Chebyshev(r_coefs)
    #     print(obj(0.1,0.2,0.3,0.4))

    def test2DConvexFunc(self):
        def g(x, y):
            return math.sin(x) + math.cos(2 * y) + x * y
        obj = cheb.Chebyshev.interpolate(np.vectorize(g), 10)
        self.assertAlmostEqual(g(.4, .5), obj(.4, .5))
        self.assertAlmostEqual(g(.5, .4), obj(.5, .4))
        # r_value = list(r("library(chebpol)\n"
        #               + "g <- function(x) sin( x[1] )  + cos( 2 * x[2] )\n"
        #               + "ch <- ipol(g,dims=c(10,10),method='cheb')\n"
        #               + "ch(c(.5,.4))") )
        # self.assertAlmostEqual(g(.5,.4),r_value[0],delta=0.000001)
        #
        # cheb_pts = cheb.chebpts1(10)[::-1]
        # num_of_arguments = 2
        # grid = [x[::-1] for x in product(cheb_pts, repeat=num_of_arguments)]
        # values = [g(*args) for args in grid]
        # values = [ values[0] ] + values
        # code = 'values <- c(' + ','.join(np.char.mod('%.10f', values)) + ')\n' \
        #        + 'make.g <- function() {\n'  \
        #        + '  i <- 0\n' \
        #        + '  g <- function(x){\n' \
        #        + '      i <<- i + 1\n' \
        #        + '      return (values[ i ])\n'\
        #        + '  }\n' \
        #        + '  return (g)\n' \
        #        + '}\n' \
        #        + 'g <- make.g()\n'
        # r_value = list(r(code
        #               + "ch <- ipol(g,dims=c(10,10),method='cheb')\n"
        #               + "ch(c(.4,.5))") )
        # self.assertAlmostEqual(g(.4,.5),r_value[0],delta=0.000001)

    def testBlackScholes(self):
        price = call_price(23.75, 15., 0.01, 0.35, 0.5)
        # See: https://www.mystockoptions.com/black-scholes.cfm?s=23.75&x=15&t=0.5&r=1%25&v=35%25&calculate=Calculate
        self.assertAlmostEqual(8.879159263714124, price, delta=0.001)

    def test1D_BlackScholes(self):
        x_r_sigma_t = [15., 0.01, 0.35,  0.5]
        obj = cheb.Chebyshev.interpolate(np.vectorize(call_price), 25, [0, 100], x_r_sigma_t)
        price = call_price(23.75, *x_r_sigma_t)
        self.assertAlmostEqual(price, obj(23.75), delta=0.001)

    def test2D_BlackScholes(self):
        f = lambda s, sigma: call_price(s, 15., 0.01, sigma, 0.5)
        obj = Chebyshev.interpolate(np.vectorize(f), 25, [[10, 100], [0.1, 1.]])
        price = obj(23.75, 0.35)
        self.assertAlmostEqual(8.879159263714124, price, delta=0.001)

