# Neccesary libraries
#{{{
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from scipy import stats
from timeit import default_timer as timer
from tqdm import tqdm

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Comment this line if want to work with GPU

from scdc.ensemble import Ensemble
from scdc.event import Event
from scdc.particle import Quasiparticle
from scdc.materials import ALUMINUM, NIOBIUM, SILICON
#from scdc import plot  # Contains matplotlib configuration code
from scdc.initial.distribution.integral import InitialSampler
from scdc.initial.halo import StandardHaloDistribution
from scdc.initial.response import HybridResponseFunction
from scdc.initial.matrix_element import FiducialMatrixElement
#}}}

# Configuration
#{{{
KMS = 3.33564e-6  # km/s in natural units
mediator_mass = 10

material = SILICON
vdf = StandardHaloDistribution(
    v_0    = 220 * KMS / material.v, 
    v_esc  = 550 * KMS / material.v,
    v_wind = 230 * KMS / material.v
)
vdf_iso = StandardHaloDistribution(
    v_0    = 220 * KMS / material.v, 
    v_esc  = 550 * KMS / material.v,
    v_wind = 0 * KMS / material.v
)
response = HybridResponseFunction(material, 1) # The 1 is the coherence sign. Can be +1 or -1
me_light = FiducialMatrixElement(mediator_mass = 0)
me_heavy = FiducialMatrixElement(mediator_mass = mediator_mass)
m_nt     = [1e6, 1e7, 1e8, 1e9, 1e10, 1e11] / material.m #np.concatenate((
           #np.linspace(1, 9, 3) * 1e4, 
           #np.linspace(1, 9, 3) * 1e5
           #)) / material.m # Dark matter masses
N_events = np.array( [100] ) # Numero de eventos observados
#}}}

# Loops
#{{{
sim_light_energy_leaf_qp = []
sim_light_energy_leaf_ph = []
sim_light_dep_energy     = []
for i, vali in enumerate(tqdm(m_nt)):
    sampler_light    = InitialSampler(vali, me_light, material, response, vdf, n_cq = 20, n_rq = 20)
    simulation_light = sampler_light.ensemble(N_events[0])
    simulation_light.chain()
    sim_light_energy_leaf_qp.append(simulation_light.leaf_particles.quasiparticles.energy)
    sim_light_energy_leaf_ph.append(simulation_light.leaf_particles.phonons.energy)

    aux = np.zeros((N_events[0]))
    for j, valj in enumerate(simulation_light):
        aux[j] = sum(valj.leaf_particles.phonons.energy) + sum(valj.leaf_particles.quasiparticles.energy)
    sim_light_dep_energy.append(aux)

sim_heavy_energy_leaf_qp = []
sim_heavy_energy_leaf_ph = []
sim_heavy_dep_energy     = []
for i, vali in enumerate(tqdm(m_nt)):
    sampler_heavy    = InitialSampler(vali, me_heavy, material, response, vdf, n_cq = 20, n_rq = 20)
    simulation_heavy = sampler_heavy.ensemble(N_events[0])
    simulation_heavy.chain()
    sim_heavy_energy_leaf_qp.append(simulation_heavy.leaf_particles.quasiparticles.energy)
    sim_heavy_energy_leaf_ph.append(simulation_heavy.leaf_particles.phonons.energy)
    
    aux = np.zeros((N_events[0]))
    for j, valj in enumerate(simulation_heavy):
        aux[j] = sum(valj.leaf_particles.phonons.energy) + sum(valj.leaf_particles.quasiparticles.energy)
    sim_heavy_dep_energy.append(aux)
#}}}

# Graphs
#{{{
cmap = get_cmap('viridis', len(m_nt))

fig, ax = plt.subplots(2, 2, sharex = False, sharey = False, figsize = (14, 10), gridspec_kw = dict(hspace = 0.3, wspace = 0))

for i, vali in enumerate(m_nt):
    if i <= (len(m_nt)/2):
        ax[0,0].hist(sim_light_energy_leaf_qp[i], histtype = 'step', color = cmap(i),
                label = 'M_{DM} = ' + '{:.2e}'.format(vali * material.m) + ' eV')
    else:
        ax[0,0].hist(sim_light_energy_leaf_qp[i], histtype = 'step', color = cmap(i))
    ax[1,0].hist(sim_light_energy_leaf_ph[i], histtype = 'step', color = cmap(i))

    
for i, vali in enumerate(m_nt):
    if i > (len(m_nt)/2):
        ax[0,1].hist(sim_heavy_energy_leaf_qp[i], histtype = 'step', color = cmap(i),
                label = 'M_{DM} = ' + '{:.2e}'.format(vali * material.m) + ' eV')
    else:
        ax[0,1].hist(sim_heavy_energy_leaf_qp[i], histtype = 'step', color = cmap(i))
    ax[1,1].hist(sim_heavy_energy_leaf_ph[i], histtype = 'step', color = cmap(i))
    
ax[0,1].legend()
ax[0,0].legend()
ax[0,0].set_title('Light Mediator m = 0')
ax[0,1].set_title('Heavy Mediator m = ' + str(mediator_mass))

ax[0,0].set_xlabel('Energy [$\Delta$]')
ax[0,1].set_xlabel('Energy [$\Delta$]')
ax[0,1].yaxis.set_ticks_position('both')
ax[0,1].yaxis.tick_right()
ax[0,0].text(1 + 0.8e-5, 65, 'QP')
ax[0,1].text(1 + 0.8e-5, 65, 'QP')
 
ax[1,0].set_xlabel('Energy [$\Delta$]')
ax[1,1].set_xlabel('Energy [$\Delta$]')
ax[1,1].yaxis.set_ticks_position('both')
ax[1,1].yaxis.tick_right()
ax[1,0].text(0.17, 700, 'PH')
ax[1,1].text(0.17, 700, 'PH')

plt.savefig('../graph/energy_leaf_QP+PH_AL_' + 
            str(np.min(m_nt * material.m)) + '-' + str(np.max(m_nt * material.m)) + 
            '_MedMass_' + str(mediator_mass) + '.pdf')

cmap = get_cmap('viridis', len(m_nt))

fig, ax = plt.subplots(1, 2, sharex = False, sharey = False, figsize = (10, 5), gridspec_kw = dict(hspace = 0.3, wspace = 0))

for i, vali in enumerate(m_nt):
    if i <= (len(m_nt)/2):
        ax[0].hist(sim_light_dep_energy[i], histtype = 'step', color = cmap(i),
                label = 'M_{DM} = ' + '{:.2e}'.format(vali * material.m) + ' eV')
    else:
        ax[0].hist(sim_light_dep_energy[i], histtype = 'step', color = cmap(i))

    
for i, vali in enumerate(m_nt):
    if i > (len(m_nt)/2):
        ax[1].hist(sim_heavy_dep_energy[i], histtype = 'step', color = cmap(i),
                label = 'M_{DM} = ' + '{:.2e}'.format(vali * material.m) + ' eV')
    else:
        ax[1].hist(sim_heavy_dep_energy[i], histtype = 'step', color = cmap(i))
    
ax[1].legend()
ax[0].legend()
ax[0].set_title('Light Mediator m = 0')
ax[1].set_title('Heavy Mediator m = 10')

ax[0].set_xlabel('Energy [$\Delta$]')
ax[1].set_xlabel('Energy [$\Delta$]')
ax[1].yaxis.set_ticks_position('both')
ax[1].yaxis.tick_right()

plt.savefig('../graph/dep_energy_AL_' + 
            str(np.min(m_nt * material.m)) + '-' + str(np.max(m_nt * material.m)) + 
            '_MedMass_' + str(mediator_mass) + '.pdf')
#}}}
#{{{
cmap = get_cmap('viridis', len(m_nt))

fig, ax = plt.subplots(2, 2, sharex = False, sharey = False, figsize = (14, 10), gridspec_kw = dict(hspace = 0.3, wspace = 0))

for i, vali in enumerate(m_nt):
    if i < (len(m_nt)/2):
        ax[0,0].hist(sim_light_energy_leaf_qp[i], histtype = 'step', density = True, color = cmap(i),
                label = 'M_{DM} = ' + '{:.2e}'.format(vali * material.m) + ' eV')
    else:
        ax[0,0].hist(sim_light_energy_leaf_qp[i], histtype = 'step', density = True, color = cmap(i))
    ax[1,0].hist(sim_light_energy_leaf_ph[i], histtype = 'step', density = True, color = cmap(i))

    
for i, vali in enumerate(m_nt):
    if i > (len(m_nt)/2):
        ax[0,1].hist(sim_heavy_energy_leaf_qp[i], histtype = 'step', density = True, color = cmap(i),
                label = 'M_{DM} = ' + '{:.2e}'.format(vali * material.m) + ' eV')
    else:
        ax[0,1].hist(sim_heavy_energy_leaf_qp[i], histtype = 'step', density = True, color = cmap(i))
    ax[1,1].hist(sim_heavy_energy_leaf_ph[i], histtype = 'step', density = True, color = cmap(i))
    
ax[0,1].legend()
ax[0,0].legend()
ax[0,0].set_title('Light Mediator m = 0')
ax[0,1].set_title('Heavy Mediator m = ' + str(mediator_mass))

ax[0,0].set_xlabel('Energy [$\Delta$]')
ax[0,1].set_xlabel('Energy [$\Delta$]')
ax[0,1].yaxis.set_ticks_position('both')
ax[0,1].yaxis.tick_right()
ax[0,0].text(1 + 0.8e-5, 65, 'QP')
ax[0,1].text(1 + 0.8e-5, 65, 'QP')
 
ax[1,0].set_xlabel('Energy [$\Delta$]')
ax[1,1].set_xlabel('Energy [$\Delta$]')
ax[1,1].yaxis.set_ticks_position('both')
ax[1,1].yaxis.tick_right()
ax[1,0].text(0.17, 700, 'PH')
ax[1,1].text(0.17, 700, 'PH')

plt.savefig('../graph/energy_leaf_QP+PH_AL_' + 
            str(np.min(m_nt * material.m)) + '-' + str(np.max(m_nt * material.m)) + 
            '_MedMass_' + str(mediator_mass) + '_density.pdf')

cmap = get_cmap('viridis', len(m_nt))

fig, ax = plt.subplots(1, 2, sharex = False, sharey = False, figsize = (10, 5), gridspec_kw = dict(hspace = 0.3, wspace = 0))

for i, vali in enumerate(m_nt):
    if i < (len(m_nt)/2):
        ax[0].hist(sim_light_dep_energy[i], histtype = 'step', density = True, color = cmap(i),
                label = 'M_{DM} = ' + '{:.2e}'.format(vali * material.m) + ' eV')
    else:
        ax[0].hist(sim_light_dep_energy[i], histtype = 'step', density = True, color = cmap(i))

    
for i, vali in enumerate(m_nt):
    if i > (len(m_nt)/2):
        ax[1].hist(sim_heavy_dep_energy[i], histtype = 'step', density = True, color = cmap(i),
                label = 'M_{DM} = ' + '{:.2e}'.format(vali * material.m) + ' eV')
    else:
        ax[1].hist(sim_heavy_dep_energy[i], histtype = 'step', density = True, color = cmap(i))
    
ax[1].legend()
ax[0].legend()
ax[0].set_title('Light Mediator m = 0')
ax[1].set_title('Heavy Mediator m = ' + str(mediator_mass))

ax[0].set_xlabel('Energy [$\Delta$]')
ax[1].set_xlabel('Energy [$\Delta$]')
ax[1].yaxis.set_ticks_position('both')
ax[1].yaxis.tick_right()

plt.savefig('../graph/dep_energy_AL_' + 
            str(np.min(m_nt * material.m)) + '-' + str(np.max(m_nt * material.m)) + 
            '_MedMass_' + str(mediator_mass) + '_density.pdf')
#}}}
