import math
import unittest
from chebyshev.Chebyshev import Chebyshev

from chebyshev.blackscholes import call_price


class ChebyshevTest(unittest.TestCase):

    def testPolynomial(self):
        def f(x):
            return x*x + 2
        # check 1d function without domains
        obj = Chebyshev.interpolate(f, 10)
        self.assertAlmostEqual(2.25, obj(.5))

        # and with domains
        obj = Chebyshev.interpolate(f, 10, domain=[0, 100])
        self.assertAlmostEqual(27, obj(5))

    def test2DPolynomial(self):
        def g(x,y):
            return x + y * 10
        # check 2d function without domains
        obj = Chebyshev.interpolate(g, 10)
        self.assertAlmostEqual(g(.2, .1), obj(.2, .1))
        # and with domains
        obj = Chebyshev.interpolate(g, 10, domain=[[-10, 10], [-100, 100]])
        self.assertAlmostEqual(g(1., 2.), obj(1., 2.))

    def test3DPolynomial(self):
        def h(x, y, z):
            return 30 * x + y * 5 + z
        # check 3d function without domains
        obj = Chebyshev.interpolate(h, 10)
        self.assertAlmostEqual(h(.2, .1,.3), obj(.2, .1,.3))
        # and with domains
        obj = Chebyshev.interpolate(h, 10, domain=[[-10, 10], [-100, 100], [-1000, 1000]])

        self.assertAlmostEqual(h(0., 3., 0.), obj(0., 3., 0.))
        self.assertAlmostEqual(h(1., 0., 0.), obj(1., 0., 0.))
        self.assertAlmostEqual(h(0., 0., 3.), obj(0., 0., 3.))
        self.assertAlmostEqual(h(1., 2., 3.), obj(1., 2., 3.))

    # def testNDimension(self):
    #     r_coefs = np.asarray(r('library(chebpol)\n'
    #                 + 'g <- function(x) x[1] + x[2]^2 + exp(x[3]) + cos(x[4])\n'
    #                 + 'chebcoef( evalongrid(g,c(5,5,5,5)) )'))
    #     obj = cheb.Chebyshev(r_coefs)
    #     print(obj(0.1,0.2,0.3,0.4))

    def test2DConvexFunc(self):
        def g(x, y):
            return math.sin(x) + math.cos(2 * y) + x * y
        #check 2d function without domains
        obj = Chebyshev.interpolate(g, 10)
        self.assertAlmostEqual(g(.4, .5), obj(.4, .5))
        self.assertAlmostEqual(g(.5, .4), obj(.5, .4))
        #and with domains
        obj = Chebyshev.interpolate(g, 18, domain=[[-math.pi, math.pi], [-math.pi, math.pi]])
        self.assertAlmostEqual(g(1.4, 1.5), obj(1.4, 1.5))
        self.assertAlmostEqual(g(1.5, 1.4), obj(1.5, 1.4))

    def testBlackScholes(self):
        price = call_price(23.75, 15., 0.01, 0.35, 0.5)
        # See: https://www.mystockoptions.com/black-scholes.cfm?s=23.75&x=15&t=0.5&r=1%25&v=35%25&calculate=Calculate
        self.assertAlmostEqual(8.879159263714124, price, delta=0.001)

    def test1D_BlackScholes(self):
        x_r_sigma_t = [15., 0.01, 0.35,  0.5]
        obj = Chebyshev.interpolate(call_price, 25, [0, 100], x_r_sigma_t)
        price = call_price(23.75, *x_r_sigma_t)
        self.assertAlmostEqual(price, obj(23.75), delta=0.001)

    def test2D_BlackScholes(self):
        f = lambda s, sigma: call_price(s, 15., 0.01, sigma, 0.5)
        obj = Chebyshev.interpolate(f, 25, [[10, 100], [0.1, 1.]])
        price = obj(23.75, 0.35)
        self.assertAlmostEqual(8.879159263714124, price, delta=0.001)

    # def testND_BlackScholes(self):
    #     obj = Chebyshev.interpolate(call_price, 25, [[10, 100], [10, 100], [0., .2], [0.1, 1.], [0.5, 5]])
    #     price = obj(23.75, 15., 0.01, 0.35,  0.5)
    #     self.assertAlmostEqual(8.879159263714124, price, delta=0.001)
