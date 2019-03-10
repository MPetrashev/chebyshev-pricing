import math


def phi(x):
    """
    Return the value of the Gaussian probability function with mean 0.0 and standard deviation 1.0 at the given x value.
    """
    return math.exp(-x * x / 2.0) / math.sqrt(2.0 * math.pi)


def pdf(x, mu=0.0, sigma=1.0):
    """
    Return the value of the Gaussian probability function with mean mu and standard deviation sigma at the given x value
    """
    return phi((x - mu) / sigma) / sigma


def Phi(z):
    """
    Return the value of the cumulative Gaussian distribution function with mean 0.0 and standard deviation 1.0 at the
    given z value.
    """
    if z < -8.0:
        return 0.0
    elif z > 8.0:
        return 1.0

    total = 0.0
    term = z
    i = 3
    while total != total + term:
        total += term
        term *= z * z / float(i)
        i += 2
    return 0.5 + total * phi(z)


def cdf(z, mu=0.0, sigma=1.0):
    """
    Return standard Gaussian cdf with mean mu and stddev sigma. Use Taylor approximation.
    """
    return Phi((z - mu) / sigma)


def call_price(s, x, r, sigma, t):
    """
    Black-Scholes formula.
    """
    a = (math.log(s/x) + (r + sigma * sigma/2.0) * t) / (sigma * math.sqrt(t))
    b = a - sigma * math.sqrt(t)
    return s * cdf(a) - x * math.exp(-r * t) * cdf(b)
