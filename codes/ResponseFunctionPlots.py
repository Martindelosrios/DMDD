# Neccesary libraries
#{{{
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.cm import get_cmap

from scdc.materials import ALUMINUM, NIOBIUM, SILICON
from scdc.initial.response import HybridResponseFunction
#}}}

material = ALUMINUM
response = HybridResponseFunction(material, 1) # The 1 is the coherence sign. Can be +1 or -1


# response is a callable with inputs 
#   r1 (float): magnitude of momentum of QP 1. 
#   r2 (float): magnitude of momentum of QP 2.
#   q (float): magnitude of total momentum transfer.
#   omega (float): energy deposit.

'''
* I think that this HybridResponseFucntion correspond to the integrand of eq. S.21 of 2109.04473.

* Inside the response function r1 and r2 goes into the self.material.coherence_uvvu function, that says that thhey must be in material units.
* Inside the response function q and w goes into the self.material.epsilon_lindhard function but there is no information of the units.
'''

# r1 and r2 fix
#{{{

w = np.logspace(-5, 1.5, 200)
q = np.logspace(-5, -4, 20)
cmap = matplotlib.cm.viridis
norm = matplotlib.colors.LogNorm(vmin=np.min(q), vmax=np.max(q))

fig, axs = plt.subplots(4, 2, sharex = True, gridspec_kw = dict(hspace = 0, wspace = 0), figsize = (12,10))

axs[0,0].set_ylabel('Response Function')
axs[0,0].axis('off')

r1 = 10
r2 = 10
for i, vali in enumerate(q):
    res = []
    for j in w:
        res.append(response(r1 = r1, r2 = r2, q = vali, omega = j).numpy())
    axs[0,1].plot(w, np.asarray(res), label = 'q = {:.2e}'.format(vali) , c = cmap(norm(vali)) )
axs[0,1].legend(loc = 'upper left', ncol = 3, bbox_to_anchor=(-1.0, 1.05))
axs[0,1].yaxis.tick_right()
axs[0,1].yaxis.set_label_position("right")
axs[0,1].set_ylabel('Response Function')
axs[0,1].set_yscale('log')
axs[0,1].text(15, 1e-9, 'r1 = {}; r2 = {}'.format(r1, r2))

r1 = 1
r2 = 1
for i, vali in enumerate(q):
    res = []
    for j in w:
        res.append(response(r1 = r1, r2 = r2, q = vali, omega = j).numpy())
    axs[1,1].plot(w, np.asarray(res), label = 'q = {:.2e}'.format(vali) , c = cmap(norm(vali)) )
axs[1,1].yaxis.tick_right()
axs[1,1].yaxis.set_label_position("right")
axs[1,1].set_ylabel('Response Function')
axs[1,1].set_yscale('log')
axs[1,1].text(15, 1e4, 'r1 = {}; r2 = {}'.format(r1, r2))

r1 = 0.1
r2 = 0.1
for i, vali in enumerate(q):
    res = []
    for j in w:
        res.append(response(r1 = r1, r2 = r2, q = vali, omega = j).numpy())
    axs[2,1].plot(w, np.asarray(res), label = 'q = {:.2e}'.format(vali) , c = cmap(norm(vali)) )
axs[2,1].yaxis.tick_right()
axs[2,1].yaxis.set_label_position("right")
axs[2,1].set_ylabel('Response Function')
axs[2,1].set_yscale('log')
axs[2,1].text(15, 1e-6, 'r1 = {}; r2 = {}'.format(r1, r2))

r1 = 1e-8
r2 = 1e-8
for i, vali in enumerate(q):
    res = []
    for j in w:
        res.append(response(r1 = r1, r2 = r2, q = vali, omega = j).numpy())
    axs[3,1].plot(w, np.asarray(res), label = 'q = {:.2e}'.format(vali) , c = cmap(norm(vali)) )
axs[3,1].yaxis.tick_right()
axs[3,1].yaxis.set_label_position("right")
axs[3,1].set_ylabel('Response Function')
axs[3,1].set_yscale('log')
axs[3,1].text(15, 1e-7, 'r1 = {}; r2 = {}'.format(r1, r2))
axs[3,1].set_xlabel('Deposited Energy')

r1 = 1
r2 = 10
for i, vali in enumerate(q):
    res = []
    for j in w:
        res.append(response(r1 = r1, r2 = r2, q = vali, omega = j).numpy())
    axs[1,0].plot(w, np.asarray(res), label = 'q = {:.2e}'.format(vali) , c = cmap(norm(vali)) )
axs[1,0].set_ylabel('Response Function')
axs[1,0].set_yscale('log')
axs[1,0].text(15, 1e4, 'r1 = {}; r2 = {}'.format(r1, r2))

r1 = 1
r2 = 0.1
for i, vali in enumerate(q):
    res = []
    for j in w:
        res.append(response(r1 = r1, r2 = r2, q = vali, omega = j).numpy())
    axs[2,0].plot(w, np.asarray(res), label = 'q = {:.2e}'.format(vali) , c = cmap(norm(vali)) )
axs[2,0].set_ylabel('Response Function')
axs[2,0].set_yscale('log')
axs[2,0].text(15, 1e4, 'r1 = {}; r2 = {}'.format(r1, r2))

r1 = 1
r2 = 1e-8
for i, vali in enumerate(q):
    res = []
    for j in w:
        res.append(response(r1 = r1, r2 = r2, q = vali, omega = j).numpy())
    axs[3,0].plot(w, np.asarray(res), label = 'q = {:.2e}'.format(vali) , c = cmap(norm(vali)) )
axs[3,0].set_ylabel('Response Function')
axs[3,0].set_xlabel('Deposited Energy')
axs[3,0].set_yscale('log')
axs[3,0].text(15, 1e4, 'r1 = {}; r2 = {}'.format(r1, r2))

plt.show()

#}}}

# w and r2 fix
#{{{

r1 = np.logspace(-3, 0.51, 200)
q  = np.logspace(-5, -0, 20)
r2 = 1

cmap = matplotlib.cm.viridis
norm = matplotlib.colors.LogNorm(vmin=np.min(q), vmax=np.max(q))

fig, axs = plt.subplots(4, 2, sharex = True, gridspec_kw = dict(hspace = 0, wspace = 0), figsize = (12,10))

axs[0,0].set_ylabel('Response Function')
axs[0,1].axis('off')

w  = 1e-2
for i, vali in enumerate(q):
    res = []
    for j in r1:
        res.append(response(r1 = j, r2 = r2, q = vali, omega = w).numpy())
    axs[0,0].plot(r1, np.asarray(res), label = 'q = {:.2e}'.format(vali) , c = cmap(norm(vali)) )
axs[0,0].legend(loc = 'upper right', ncol = 3, bbox_to_anchor=(2.0, 1.05))
axs[0,0].set_ylabel('Response Function')
axs[0,0].set_yscale('log')
axs[0,0].text(1.5, 1e-1, 'w = {}; r2 = {}'.format(w, r2))

w  = 1e-1
for i, vali in enumerate(q):
    res = []
    for j in r1:
        res.append(response(r1 = j, r2 = r2, q = vali, omega = w).numpy())
    axs[1,0].plot(r1, np.asarray(res), label = 'q = {:.2e}'.format(vali) , c = cmap(norm(vali)) )
axs[1,0].set_ylabel('Response Function')
axs[1,0].set_yscale('log')
axs[1,0].text(1.5, 1e-1, 'w = {}; r2 = {}'.format(w, r2))

w  = 0
for i, vali in enumerate(q):
    res = []
    for j in r1:
        res.append(response(r1 = j, r2 = r2, q = vali, omega = w).numpy())
    axs[2,0].plot(r1, np.asarray(res), label = 'q = {:.2e}'.format(vali) , c = cmap(norm(vali)) )
axs[2,0].set_ylabel('Response Function')
axs[2,0].set_yscale('log')
axs[2,0].text(1.5, 1e-1, 'w = {}; r2 = {}'.format(w, r2))

w  = 1
for i, vali in enumerate(q):
    res = []
    for j in r1:
        res.append(response(r1 = j, r2 = r2, q = vali, omega = w).numpy())
    axs[3,0].plot(r1, np.asarray(res), label = 'q = {:.2e}'.format(vali) , c = cmap(norm(vali)) )
axs[3,0].set_ylabel('Response Function')
axs[3,0].set_xlabel('$r_{1}$')
axs[3,0].set_yscale('log')
axs[3,0].text(1.5, 1e-1, 'w = {}; r2 = {}'.format(w, r2))


w  = 15
for i, vali in enumerate(q):
    res = []
    for j in r1:
        res.append(response(r1 = j, r2 = r2, q = vali, omega = w).numpy())
    axs[1,1].plot(r1, np.asarray(res), label = 'q = {:.2e}'.format(vali) , c = cmap(norm(vali)) )
axs[1,1].yaxis.tick_right()
axs[1,1].yaxis.set_label_position("right")
axs[1,1].set_ylabel('Response Function')
axs[1,1].set_yscale('log')
axs[1,1].text(1.5, 2, 'w = {}; r2 = {}'.format(w, r2))

w  = 50
for i, vali in enumerate(q):
    res = []
    for j in r1:
        res.append(response(r1 = j, r2 = r2, q = vali, omega = w).numpy())
    axs[2,1].plot(r1, np.asarray(res), label = 'q = {:.2e}'.format(vali) , c = cmap(norm(vali)) )
axs[2,1].yaxis.tick_right()
axs[2,1].yaxis.set_label_position("right")
axs[2,1].set_ylabel('Response Function')
axs[2,1].set_yscale('log')
axs[2,1].text(1.5, 1, 'w = {}; r2 = {}'.format(w, r2))

w  = 100
for i, vali in enumerate(q):
    res = []
    for j in r1:
        res.append(response(r1 = j, r2 = r2, q = vali, omega = w).numpy())
    axs[3,1].plot(r1, np.asarray(res), label = 'q = {:.2e}'.format(vali) , c = cmap(norm(vali)) )
axs[3,1].yaxis.tick_right()
axs[3,1].yaxis.set_label_position("right")
axs[3,1].set_ylabel('Response Function')
axs[3,1].set_yscale('log')
axs[3,1].text(1.5, 5e-1, 'w = {}; r2 = {}'.format(w, r2))
axs[3,1].set_xlabel('$r_{1}$')

plt.show()

#}}}

