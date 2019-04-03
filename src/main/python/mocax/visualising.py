from chebyshev.blackscholes import call_price
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import mocaxpy

my_function = np.vectorize(lambda x: call_price(x, 100, 0.05, 0.4, 1.))

num_dimensions = 1
a = 30
b = 150

N = 3

domain = mocaxpy.MocaxDomain([[a, b]])
ns = mocaxpy.MocaxNs([N])
obj = mocaxpy.Mocax(None, num_dimensions, domain, None, ns)
anchor_vals = my_function(np.array(obj.get_evaluation_points()).flatten())
obj.set_original_function_values( list(anchor_vals) )

pts = np.linspace(a, b, 500)

plt.figure(figsize=(12,4))
plt.plot(pts, my_function(pts), label='Function values')
plt.plot(pts, np.vectorize( obj.eval )(pts), label='Chebyshev values')
plt.legend()

plt.show()