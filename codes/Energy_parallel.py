# Neccesary libraries
#{{{
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from scipy import stats
from timeit import default_timer as timer
from tqdm import tqdm
from joblib import Parallel, delayed
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

# Custom functions
#{{{
def analyse(i, matrix_element):
    sampler    = InitialSampler(m_nt[i], matrix_element, material, response, vdf, n_cq = 20, n_rq = 20)
    simulation = sampler.ensemble(N_events[0])
    simulation.chain()
    aux = np.zeros((N_events[0]))
    for j, valj in enumerate(simulation):
        aux[j] = sum(valj.leaf_particles.phonons.energy) + sum(valj.leaf_particles.quasiparticles.energy)
    return [simulation.leaf_particles.quasiparticles.energy, simulation.leaf_particles.phonons.energy, aux]


#}}}

# Configuration
#{{{
KMS = 3.33564e-6  # km/s in natural units
mediator_mass = 100

material = ALUMINUM
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
m_nt     = [1e7, 1e8] / material.m #np.concatenate((
           #np.linspace(1, 9, 3) * 1e4, 
           #np.linspace(1, 9, 3) * 1e5
           #)) / material.m # Dark matter masses
N_events = np.array( [100] ) # Numero de eventos observados
#}}}

print('\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\')
print('\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\')
print('Starting analysis with light mediator')
print('\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\')
print('\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\')
results_light = Parallel(n_jobs=1)(delayed(analyse)(i, matrix_element = me_light) for i in tqdm(range(len(m_nt))))
print('\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\')
print('\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\')
print('Starting analysis with heavy mediator')
print('\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\')
print('\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\')
results_heavy = Parallel(n_jobs=1)(delayed(analyse)(i, matrix_element = me_heavy) for i in tqdm(range(len(m_nt))))

# Graphs
#{{{
cmap = get_cmap('viridis', len(m_nt))

fig, ax = plt.subplots(2, 2, sharex = False, sharey = False, figsize = (14, 10), gridspec_kw = dict(hspace = 0.3, wspace = 0))

for i, vali in enumerate(m_nt):
    if i <= (len(m_nt)/2):
        ax[0,0].hist(results_light[i][0], histtype = 'step', color = cmap(i),
                label = 'M_{DM} = ' + '{:.2e}'.format(vali * material.m) + ' eV')
    else:
        ax[0,0].hist(results_light[i][0], histtype = 'step', color = cmap(i))
    ax[1,0].hist(results_light[i][1], histtype = 'step', color = cmap(i))

    
for i, vali in enumerate(m_nt):
    if i > (len(m_nt)/2):
        ax[0,1].hist(results_heavy[i][0], histtype = 'step', color = cmap(i),
                label = 'M_{DM} = ' + '{:.2e}'.format(vali * material.m) + ' eV')
    else:
        ax[0,1].hist(results_heavy[i][0], histtype = 'step', color = cmap(i))
    ax[1,1].hist(results_heavy[i][1], histtype = 'step', color = cmap(i))
    
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
            #str(np.min(m_nt * material.m)) + '-' + str(np.max(m_nt * material.m)) + 
            "{:.2e}".format(np.min(m_nt * material.m)) + "-{:.2e}".format(np.max(m_nt * material.m)) +
            '_MedMass_' + str(mediator_mass) + '_p.pdf')

cmap = get_cmap('viridis', len(m_nt))

fig, ax = plt.subplots(1, 2, sharex = False, sharey = False, figsize = (10, 5), gridspec_kw = dict(hspace = 0.3, wspace = 0))

for i, vali in enumerate(m_nt):
    if i <= (len(m_nt)/2):
        ax[0].hist(results_light[i][2], histtype = 'step', color = cmap(i),
                label = 'M_{DM} = ' + '{:.2e}'.format(vali * material.m) + ' eV')
    else:
        ax[0].hist(results_light[i][2], histtype = 'step', color = cmap(i))

    
for i, vali in enumerate(m_nt):
    if i > (len(m_nt)/2):
        ax[1].hist(results_heavy[i][2], histtype = 'step', color = cmap(i),
                label = 'M_{DM} = ' + '-{:.2e}'.format(vali * material.m) + ' eV')
    else:
        ax[1].hist(results_heavy[i][2], histtype = 'step', color = cmap(i))
    
ax[1].legend()
ax[0].legend()
ax[0].set_title('Light Mediator m = 0')
ax[1].set_title('Heavy Mediator m = 10')

ax[0].set_xlabel('Energy [$\Delta$]')
ax[1].set_xlabel('Energy [$\Delta$]')
ax[1].yaxis.set_ticks_position('both')
ax[1].yaxis.tick_right()

plt.savefig('../graph/dep_energy_AL_' + 
            #str(np.min(m_nt * material.m)) + '-' + str(np.max(m_nt * material.m)) + 
            "{:.2e}".format(np.min(m_nt * material.m)) + "{:.2e}".format(np.max(m_nt * material.m)) +
            '_MedMass_' + str(mediator_mass) + '_p.pdf')
#}}}
#{{{
cmap = get_cmap('viridis', len(m_nt))

fig, ax = plt.subplots(2, 2, sharex = False, sharey = False, figsize = (14, 10), gridspec_kw = dict(hspace = 0.3, wspace = 0))

for i, vali in enumerate(m_nt):
    if i <= (len(m_nt)/2):
        ax[0,0].hist(results_light[i][0], histtype = 'step', density = True, color = cmap(i),
                label = 'M_{DM} = ' + '{:.2e}'.format(vali * material.m) + ' eV')
    else:
        ax[0,0].hist(results_light[i][0], histtype = 'step', density = True, color = cmap(i))
    ax[1,0].hist(results_light[i][1], histtype = 'step', density = True, color = cmap(i))

    
for i, vali in enumerate(m_nt):
    if i > (len(m_nt)/2):
        ax[0,1].hist(results_heavy[i][0], histtype = 'step', density = True, color = cmap(i),
                label = 'M_{DM} = ' + '{:.2e}'.format(vali * material.m) + ' eV')
    else:
        ax[0,1].hist(results_heavy[i][0], histtype = 'step', density = True, color = cmap(i))
    ax[1,1].hist(results_heavy[i][1], histtype = 'step', density = True, color = cmap(i))
    
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
            #str(np.min(m_nt * material.m)) + '-' + str(np.max(m_nt * material.m)) + 
            "{:.2e}".format(np.min(m_nt * material.m)) + "-{:.2e}".format(np.max(m_nt * material.m)) +
            '_MedMass_' + str(mediator_mass) + '_density_p.pdf')

cmap = get_cmap('viridis', len(m_nt))

fig, ax = plt.subplots(1, 2, sharex = False, sharey = False, figsize = (10, 5), gridspec_kw = dict(hspace = 0.3, wspace = 0))

for i, vali in enumerate(m_nt):
    if i <= (len(m_nt)/2):
        ax[0].hist(results_light[i][2], histtype = 'step', density = True, color = cmap(i),
                label = 'M_{DM} = ' + '{:.2e}'.format(vali * material.m) + ' eV')
    else:
        ax[0].hist(results_light[i][2], histtype = 'step', density = True, color = cmap(i))

    
for i, vali in enumerate(m_nt):
    if i > (len(m_nt)/2):
        ax[1].hist(results_heavy[i][2], histtype = 'step', density = True, color = cmap(i),
                label = 'M_{DM} = ' + '-{:.2e}'.format(vali * material.m) + ' eV')
    else:
        ax[1].hist(results_heavy[i][2], histtype = 'step', density = True, color = cmap(i))
    
ax[1].legend()
ax[0].legend()
ax[0].set_title('Light Mediator m = 0')
ax[1].set_title('Heavy Mediator m = ' + str(mediator_mass))

ax[0].set_xlabel('Energy [$\Delta$]')
ax[1].set_xlabel('Energy [$\Delta$]')
ax[1].yaxis.set_ticks_position('both')
ax[1].yaxis.tick_right()

plt.savefig('../graph/dep_energy_AL_' + 
            #str(np.min(m_nt * material.m)) + '-' + str(np.max(m_nt * material.m)) + 
            "{:.2e}".format(np.min(m_nt * material.m)) + "{:.2e}".format(np.max(m_nt * material.m)) +
            '_MedMass_' + str(mediator_mass) + '_density_p.pdf')
#}}}
