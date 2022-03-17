# Neccesary libraries
#{{{
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D
import seaborn as sns
import scipy.stats as st 
from sklearn.model_selection import ParameterGrid

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

# r1 and r2 fix (GRAPH)
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

# w and r2 fix (GRAPH)
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
vdf = StandardHaloDistribution(
    v_0    = 220 * KMS / material.v, 
    v_esc  = 550 * KMS / material.v,
    v_wind = 230 * KMS / material.v
)

me_heavy = FiducialMatrixElement(mediator_mass = 10)
m_dm = 1e3/material.m # Dark matter mass in material units

# Let's compute a sampling
resolution = 20
sampler = InitialSampler(m_dm, me_heavy, material, response, vdf, n_cq = resolution,
                         n_rq = resolution)

# rate grid in q and cq for fix r1 (GRAPH)
#{{{
ind_ticks = np.linspace(0, (resolution - 1), 10).astype('int')
wvals = np.logspace(0,6,10) # Multiples of Delta
cmap = matplotlib.cm.Purples
norm = matplotlib.colors.LogNorm(vmin=np.min(wvals), vmax=np.max(wvals))

custom_lines = []
for i in wvals:
    custom_lines.append( Line2D([0],[0], marker = '', linewidth = 3, linestyle = '--', color = cmap(norm(i)), 
            label = 'w = {:.2e}'.format(i)) )

plt.rc('font', size=13)  
fig, ax = plt.subplots(3,2, sharex = True, sharey = True, gridspec_kw = {'hspace':0, 'wspace':0.},
                       figsize = (14,10))

r1_vals = np.linspace(np.min(sampler.r1_vals), np.max(sampler.r1_vals), 6)

c  = 0 
for i in range(3):
    for j in range(2):
        r1 = r1_vals[c]
        c = c+1
        q_rate_grid = sampler.q_rate_grid(r1)
        extent = [ np.min(q_rate_grid[1]), np.max(q_rate_grid[1]),
                   np.min(q_rate_grid[0]), np.max(q_rate_grid[0]) ]
        
        if j == 0:
            ax[i,j].imshow(np.log10(q_rate_grid[2]), extent = extent, origin = 'lower', aspect = 'auto')
        else:
            pos = ax[i,j].imshow(np.log10(q_rate_grid[2]), extent = extent, origin = 'lower', aspect = 'auto')
            fig.colorbar(pos, ax = ax[i,j])
        rq = q_rate_grid[1][ind_ticks]
        for w in wvals:
            cq_kin_lim = (rq**2) + (w * material.Delta * 2*sampler.m1)
            cq_kin_lim = cq_kin_lim / (2*r1*rq) # Kinematic limit for omega > 0
            ax[i,j].plot(q_rate_grid[1][ind_ticks], cq_kin_lim, linestyle = '--', color = cmap(norm(w)), linewidth = 3)

        w = 2 * material.Delta
        cq_kin_lim = (rq**2) + (w*2*sampler.m1)
        cq_kin_lim = cq_kin_lim / (2*r1*rq) # Kinematic limit for omega > 0
        ax[i,j].plot(q_rate_grid[1][ind_ticks], cq_kin_lim, linestyle = '--', color = 'magenta', linewidth = 3)
        
        w = 0
        cq_kin_lim = (rq**2) + (w*2*sampler.m1)
        cq_kin_lim = cq_kin_lim / (2*r1*rq) # Kinematic limit for omega > 0
        ax[i,j].plot(q_rate_grid[1][ind_ticks], cq_kin_lim, linestyle = '--', color = 'red', linewidth = 3)

        ax[i,j].grid(True)
        ax[i,j].set_ylim(0,1)
        if i == 2:
            ax[i,j].set_xlabel('$R_{q}$')
        else:
            ax[i,j].set_xlabel('')
        if j == 0:
            ax[i,j].set_ylabel('$C_{q}$')
        else:
            ax[i,j].set_ylabel('')
        ax[i,j].text(1e-3, .2, 'r1 = {:.5f}'.format(r1))
        
        # Equations labels
        ax[i,j].text(0.0007, 0.37, '$w=0 ; cq = 2Rq/r1$', c = 'red')
        ax[0,0].text(1e-3, 0.1, '$r1 = 2 \sqrt{\Delta m_{\chi}}$')

ax[0,0].legend(handles = custom_lines, ncol = 5, bbox_to_anchor = (0.,1.3), loc = 'upper left')
plt.show()
#}}}

# rate grid in q and w for fix r1 (GRAPH)
#{{{
ind_ticks = np.linspace(0, (resolution - 1), 10).astype('int')
npoints = 20

cqvals = np.linspace(0,1,10) 
cmap = matplotlib.cm.Purples
norm = matplotlib.colors.Normalize(vmin=np.min(cqvals), vmax=np.max(cqvals))

custom_lines = []
for i in cqvals:
    custom_lines.append( Line2D([0],[0], marker = '', linewidth = 3, linestyle = '--', color = cmap(norm(i)), 
            label = 'cq = {:.2e}'.format(i)) )

plt.rc('font', size=13)  
fig, ax = plt.subplots(3,2, sharex=True, sharey=True, gridspec_kw = {'hspace':0, 'wspace':0.},
                       figsize = (14,10))

r1_vals = np.linspace(np.min(sampler.r1_vals), np.max(sampler.r1_vals), 6)

c  = 0 
for i in range(3):
    for j in range(2):
        r1 = r1_vals[c]
        c = c+1
        q_rate_grid = sampler.q_rate_grid(r1)
        
        rq_vals = np.linspace( np.min(q_rate_grid[1]), np.max(q_rate_grid[1]), npoints)
        w_vals  = np.linspace(0, ((r1**2)/ (2*sampler.m1)), npoints) #sampler._omega(r1,rq_vals, cq_vals)

        rates = np.zeros((npoints, npoints))
        for ki in range(npoints):
            w = w_vals[ki]
            for kj in range(npoints):
                rq  = rq_vals[kj]
                cq = ( (rq**2) + 2 * w * sampler.m1 ) / (2 * r1 * rq)
                if np.abs(cq) <= 1: rates[ki,kj] =  sampler.q_rate(r1, rq, cq)

        if j == 0:
            ax[i,j].imshow(np.log10(rates), origin = 'lower', extent = [np.min(rq_vals), np.max(rq_vals), 0, np.max(w_vals)], aspect = 'auto')
            ax[i,j].set_ylabel('$\omega$')
        else:
            pos = ax[i,j].imshow(np.log10(rates), origin = 'lower', extent = [np.min(rq_vals), np.max(rq_vals), 0, np.max(w_vals)], aspect = 'auto')
            fig.colorbar(pos, ax = ax[i,j])

        wlim = rq_vals * (r1 - (rq_vals/2)) / sampler.m1
        ax[i,j].plot(rq_vals, wlim)
        for cq in cqvals:
            wlim = (2 * r1 * rq_vals * cq - (rq_vals**2)) / (2*sampler.m1)
            ax[i,j].plot(rq_vals, wlim, c = cmap(norm(cq)), linestyle = '--', linewidth = 3)
        ax[i,j].text(1e-4, 17, 'r1 = {:.5f}'.format(r1))
        ax[i,j].grid(True)
        ax[i,j].set_ylim(0,np.max(wlim))
        ax[2,0].set_xlabel('rq')
        ax[2,1].set_xlabel('rq')

        ax[i,j].text(0.0008, 15, '$\omega = r_{q}(r_{1} - r_{q}/2)/2$', c = 'blue')
plt.show()
#}}}

# PDF in r3 for fix r1, rq and cq (GRAPH)
#{{{
r1_vals = np.linspace(np.min(sampler.r1_vals), np.max(sampler.r1_vals), 6)

r1 = r1_vals[3] # We will have the same r1 for all the panels
q_rate_grid = sampler.q_rate_grid(r1)

rq_vals = np.linspace(np.min(q_rate_grid[1]), np.max(q_rate_grid[1]), 6)
cq_vals = np.linspace(np.min(q_rate_grid[0]), np.max(q_rate_grid[0]), 16)

cmap = matplotlib.cm.viridis
norm = matplotlib.colors.Normalize(vmin=np.min(cq_vals), vmax=np.max(cq_vals))
fig, ax = plt.subplots(3,2, gridspec_kw = {'hspace':0.2, 'wspace':0.2},
                       figsize = (14,10))

custom_lines = []
for i in cq_vals:
    custom_lines.append( Line2D([0],[0], marker = '.', color = cmap(norm(i)), 
            label = 'Cq = {:.2f}'.format(i)) )

c  = 0 
for i in range(3):
    for j in range(2):
        rq = rq_vals[c] # Each panel will have a different rq
        c = c+1
        for cq in cq_vals:
            r2 = np.sqrt(rq**2 + r1**2 - 2*rq*r1*cq)
            omega = sampler._omega(r1, rq, cq)
            if omega >= 0:
                try:
                    r3_domain = sampler.r3_domain(r1, rq, cq)
                    cq3_domain_0 = [sampler._cq3(r1, r2, r3_domain[0], rq,  1), sampler._cq3(r1, r2, r3_domain[1], rq,  1)]
                    cq3_domain_1 = [sampler._cq3(r1, r2, r3_domain[0], rq,  -1), sampler._cq3(r1, r2, r3_domain[1], rq,  -1)]

                    r3_vals_0  = np.linspace(r3_domain[0], r3_domain[1], 100)
                    cq3_vals_0 = sampler._cq3(r1, r2, r3_vals_0, rq,  1) # We choose the + solution
                    r3_vals_0  = r3_vals_0[np.where(np.imag(cq3_vals_0) == 0)[0]] # Keep only those r3 vals that give place to real cq3 values
                    cq3_vals_0 = cq3_vals_0[np.where(np.imag(cq3_vals_0) == 0)[0]] # Keep only those r3 vals that give place to real cq3 values

                    r3_vals_1  = np.linspace(r3_domain[0], r3_domain[1], 1000)
                    cq3_vals_1 = sampler._cq3(r1, r2, r3_vals_1, rq, -1) # We choose the - solution
                    r3_vals_1  = r3_vals_1[np.where(np.imag(cq3_vals_1) == 0)[0]] # Keep only those r3 vals that give place to real cq3 values
                    cq3_vals_1 = cq3_vals_1[np.where(np.imag(cq3_vals_1) == 0)[0]] # Keep only those r3 vals that give place to real cq3 values
                    
                    r3_vals  = np.concatenate((r3_vals_0, r3_vals_1))
                    cq3_vals = np.concatenate((cq3_vals_0, cq3_vals_1))

                    ind_sort = np.argsort(r3_vals)
                    r3_vals  = r3_vals[ind_sort]
                    cq3_vals = cq3_vals[ind_sort]

                    probs = sampler.pdf(r1,rq,cq,r3_vals).numpy()
                    ax[i,j].scatter(cq3_vals, probs, c = cmap(norm(cq)) )
                    ax[i,j].axvline(x = cq3_domain_0[0], c = cmap(norm(cq)))
                    ax[i,j].axvline(x = cq3_domain_0[1], c = cmap(norm(cq)))
                    ax[i,j].axvline(x = cq3_domain_1[0], c = cmap(norm(cq)))
                    ax[i,j].axvline(x = cq3_domain_1[1], c = cmap(norm(cq)))
                except:
                    pass

        ax[i,j].text(0.05,0.9, 'rq = {:.2e}'.format(rq), transform = ax[i,j].transAxes)
        ax[i,j].set_yscale('log')
        #ax[i,j].set_ylim(7e-7, 1e-4)
ax[2,0].set_xlabel('cq3')
ax[2,1].set_xlabel('cq3')
ax[0,0].set_ylabel('Rate')
ax[1,0].set_ylabel('Rate')
ax[2,0].set_ylabel('Rate')
ax[0,0].legend(handles = custom_lines, ncol = 6, bbox_to_anchor = (2,1.4))
ax[0,0].text(0.05, 1.1, 'r1 = {:.2e}'.format(r1), transform = ax[0,0].transAxes)
plt.show()
#}}}

# Response Function in r3 for fix r1, rq and cq (GRAPH)
#{{{
r1_vals = np.linspace(np.min(sampler.r1_vals), np.max(sampler.r1_vals), 6)

r1 = r1_vals[3] # We will have the same r1 for all the panels
q_rate_grid = sampler.q_rate_grid(r1)

rq_vals = np.linspace(np.min(q_rate_grid[1]), np.max(q_rate_grid[1]), 6)
cq_vals = np.linspace(np.min(q_rate_grid[0]), np.max(q_rate_grid[0]), 16)

cmap = matplotlib.cm.viridis
norm = matplotlib.colors.Normalize(vmin=np.min(cq_vals), vmax=np.max(cq_vals))
fig, ax = plt.subplots(3,2, gridspec_kw = {'hspace':0.2, 'wspace':0.2},
                       figsize = (14,14))

custom_lines = []
for i in cq_vals:
    custom_lines.append( Line2D([0],[0], marker = '.', color = cmap(norm(i)), 
            label = 'Cq = {:.2f}'.format(i)) )

c  = 0 
for i in range(3):
    for j in range(2):
        rq = rq_vals[c] # Each panel will have a different rq
        c = c+1
        for cq in cq_vals:
            r2 = np.sqrt(rq**2 + r1**2 - 2*rq*r1*cq)
            omega = sampler._omega(r1, rq, cq)
            if omega >= 0:
                try:
                    r3_domain = sampler.r3_domain(r1, rq, cq)
                    cq3_domain_0 = [sampler._cq3(r1, r2, r3_domain[0], rq,  1), sampler._cq3(r1, r2, r3_domain[1], rq,  1)]
                    cq3_domain_1 = [sampler._cq3(r1, r2, r3_domain[0], rq,  -1), sampler._cq3(r1, r2, r3_domain[1], rq,  -1)]

                    r3_vals_0  = np.linspace(r3_domain[0], r3_domain[1], 100)
                    cq3_vals_0 = sampler._cq3(r1, r2, r3_vals_0, rq,  1) # We choose the + solution
                    r3_vals_0  = r3_vals_0[np.where(np.imag(cq3_vals_0) == 0)[0]] # Keep only those r3 vals that give place to real cq3 values
                    cq3_vals_0 = cq3_vals_0[np.where(np.imag(cq3_vals_0) == 0)[0]] # Keep only those r3 vals that give place to real cq3 values

                    r3_vals_1  = np.linspace(r3_domain[0], r3_domain[1], 1000)
                    cq3_vals_1 = sampler._cq3(r1, r2, r3_vals_1, rq, -1) # We choose the - solution
                    r3_vals_1  = r3_vals_1[np.where(np.imag(cq3_vals_1) == 0)[0]] # Keep only those r3 vals that give place to real cq3 values
                    cq3_vals_1 = cq3_vals_1[np.where(np.imag(cq3_vals_1) == 0)[0]] # Keep only those r3 vals that give place to real cq3 values
                    
                    r3_vals  = np.concatenate((r3_vals_0, r3_vals_1))
                    cq3_vals = np.concatenate((cq3_vals_0, cq3_vals_1))

                    ind_sort = np.argsort(r3_vals)
                    r3_vals  = r3_vals[ind_sort]
                    cq3_vals = cq3_vals[ind_sort]

                    r4_vals = np.sqrt(rq**2 + r3_vals**2 - 2*rq*r3_vals*cq3_vals + 0j)
                    probs = sampler.response(r3_vals, r4_vals, rq, omega).numpy()

                    ax[i,j].scatter(cq3_vals, probs, marker = '.', c = cmap(norm(cq)) )
                    ax[i,j].axvline(x = cq3_domain_0[0], c = cmap(norm(cq)))
                    ax[i,j].axvline(x = cq3_domain_0[1], c = cmap(norm(cq)))
                    ax[i,j].axvline(x = cq3_domain_1[0], c = cmap(norm(cq)))
                    ax[i,j].axvline(x = cq3_domain_1[1], c = cmap(norm(cq)))
                except:
                    pass

        ax[i,j].text(0.05,0.1, 'rq = {:.2e}'.format(rq), transform = ax[i,j].transAxes)
        ax[i,j].set_yscale('log')
        #ax[i,j].set_ylim(7e-7, 1e-4)
ax[2,0].set_xlabel('cq3')
ax[2,1].set_xlabel('cq3')
ax[0,0].set_ylabel('Rate')
ax[1,0].set_ylabel('Rate')
ax[2,0].set_ylabel('Rate')
ax[0,0].legend(handles = custom_lines, ncol = 7, bbox_to_anchor = (2,1.5))
ax[0,0].text(0.05, 1.05, 'r1 = {:.2e}'.format(r1), transform = ax[0,0].transAxes)
plt.show()
#}}}

simulation = sampler.ensemble(500)
simulation.chain()
