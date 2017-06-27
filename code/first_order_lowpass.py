import numpy as np
import matplotlib.pyplot as plt


tfn = lambda f, s: 1./np.power(1. + np.power((f/s), 2.), 0.5)

f = np.logspace(-6, -1, 101)
s = np.power(10., -3.)
pwr = 1.
fig = plt.figure()
ax = [fig.add_subplot(2, 1, i+1) for i in range(2)]
ax[0].semilogx(f, tfn(f, s))
ax[0].set_xlim(1.e-4, 1.e-2)

ax[1].loglog(f, 4.2*tfn(f, s))
ax[1].set_ylim(3.e-1, 100)
ax[1].set_xlim(1.e-6, 1.e-1)
#plt.semilogx(f, np.power(tfn(f, s), pwr))
plt.show()
