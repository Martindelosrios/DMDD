# Neccesary libraries
#{{{
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.cm import get_cmap

from scdc.materials import ALUMINUM, NIOBIUM, SILICON
from scdc.initial.response import HybridResponseFunction
from scdc.initial.distribution.integral import InitialSampler
from scdc.initial.halo import StandardHaloDistribution
from scdc.initial.response import HybridResponseFunction
from scdc.initial.matrix_element import FiducialMatrixElement
KMS = 3.33564e-6  # km/s in natural units

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


# Now let's add DM 
#{{{
vdf = StandardHaloDistribution(
    v_0    = 220 * KMS / material.v, 
    v_esc  = 550 * KMS / material.v,
    v_wind = 230 * KMS / material.v
)

me_heavy = FiducialMatrixElement(mediator_mass = 10)
m_dm = 1e3/material.m # Dark matter mass in material units
#}}}

# Let's compute a sampling
#{{{
sampler = InitialSampler(m_dm, me_heavy, material, response, vdf, n_cq = 20, n_rq = 20)

fig, ax = plt.subplots(3,2, gridspec_kw = {'hspace':0.2, 'wspace':0.2})

r1_vals = np.sort(np.random.choice(sampler.r1_vals, 6))
c  = 0 
for i in range(3):
    for j in range(2):
        r1 = r1_vals[c]
        c = c+1
        q_rate_grid = sampler.q_rate_grid(r1)
        
        sns.heatmap(np.log10(q_grid[2]), ax = ax[0,0])
        ax[i,j].set_xticks([0,10,20,30,40,50,60,70,80,90,99])
        ax[i,j].set_xticklabels(q_grid[0][[0,10,20,30,40,50,60,70,80,90,99]].round(2))
        ax[i,j].set_yticks([0,10,20,30,40,50,60,70,80,90,99])
        ax[i,j].set_yticklabels(q_grid[1][[0,10,20,30,40,50,60,70,80,90,99]].round(2))
        ax[i,j].set_xlabel('Cq')
        ax[i,j].set_ylabel('Rq')
        ax[i,j].set_title('r1 = {}'.format(r1))

r1 = 0.1
q_rate_grid = sampler.q_rate_grid(r1)

sns.heatmap(np.log10(q_grid[2]), ax = ax[1,0])
ax[1,0].set_xticks([0,10,20,30,40,50,60,70,80,90,99])
ax[1,0].set_xticklabels(q_grid[0][[0,10,20,30,40,50,60,70,80,90,99]].round(2))
ax[1,0].set_yticks([0,10,20,30,40,50,60,70,80,90,99])
ax[1,0].set_yticklabels(q_grid[1][[0,10,20,30,40,50,60,70,80,90,99]].round(2))
ax[1,0].set_xlabel('Cq')
ax[1,0].set_ylabel('Rq')
ax[1,0].set_title('r1 = {}'.format(r1))

r1 = 1
q_rate_grid = sampler.q_rate_grid(r1)

sns.heatmap(np.log10(q_grid[2]), ax = ax[2,0])
ax[2,0].set_xticks([0,10,20,30,40,50,60,70,80,90,99])
ax[2,0].set_xticklabels(q_grid[0][[0,10,20,30,40,50,60,70,80,90,99]].round(2))
ax[2,0].set_yticks([0,10,20,30,40,50,60,70,80,90,99])
ax[2,0].set_yticklabels(q_grid[1][[0,10,20,30,40,50,60,70,80,90,99]].round(2))
ax[2,0].set_xlabel('Cq')
ax[2,0].set_ylabel('Rq')
ax[2,0].set_title('r1 = {}'.format(r1))

r1 = 10
q_rate_grid = sampler.q_rate_grid(r1)

sns.heatmap(np.log10(q_grid[2]), ax = ax[0,1])
ax[0,1].set_xticks([0,10,20,30,40,50,60,70,80,90,99])
ax[0,1].set_xticklabels(q_grid[0][[0,10,20,30,40,50,60,70,80,90,99]].round(2))
ax[0,1].set_yticks([0,10,20,30,40,50,60,70,80,90,99])
ax[0,1].set_yticklabels(q_grid[1][[0,10,20,30,40,50,60,70,80,90,99]].round(2))
ax[0,1].set_xlabel('Cq')
ax[0,1].set_ylabel('Rq')
ax[0,1].set_title('r1 = {}'.format(r1))

r1 = 100
q_rate_grid = sampler.q_rate_grid(r1)

sns.heatmap(np.log10(q_grid[2]), ax = ax[1,1])
ax[1,1].set_xticks([0,10,20,30,40,50,60,70,80,90,99])
ax[1,1].set_xticklabels(q_grid[0][[0,10,20,30,40,50,60,70,80,90,99]].round(2))
ax[1,1].set_yticks([0,10,20,30,40,50,60,70,80,90,99])
ax[1,1].set_yticklabels(q_grid[1][[0,10,20,30,40,50,60,70,80,90,99]].round(2))
ax[1,1].set_xlabel('Cq')
ax[1,1].set_ylabel('Rq')
ax[1,1].set_title('r1 = {}'.format(r1))

r1 = 1000
q_rate_grid = sampler.q_rate_grid(r1)

sns.heatmap(np.log10(q_grid[2]), ax = ax[2,1])
ax[2,1].set_xticks([0,10,20,30,40,50,60,70,80,90,99])
ax[2,1].set_xticklabels(q_grid[0][[0,10,20,30,40,50,60,70,80,90,99]].round(2))
ax[2,1].set_yticks([0,10,20,30,40,50,60,70,80,90,99])
ax[2,1].set_yticklabels(q_grid[1][[0,10,20,30,40,50,60,70,80,90,99]].round(2))
ax[2,1].set_xlabel('Cq')
ax[2,1].set_ylabel('Rq')
ax[2,1].set_title('r1 = {}'.format(r1))

plt.show()

#}}}


simulation = sampler.ensemble(500)
simulation.chain()
