# Neccesary libraries
#{{{
import h5py
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
    return [simulation.leaf_particles.quasiparticles.energy, simulation.leaf_particles.phonons.energy, aux, simulation.leaf_particles.quasiparticles.cos_theta, simulation.leaf_particles.phonons.cos_theta]


# Graphs function
#{{{
def make_graph(m_nt, results, mediator_mass, mat_name):
    cmap = get_cmap('viridis', len(m_nt))
    
    fig, ax = plt.subplots(1, 2, sharex = False, sharey = False, figsize = (14, 10), gridspec_kw = dict(hspace = 0.3, wspace = 0))
    
    for i, vali in enumerate(m_nt):
        if i <= (len(m_nt)/2):
            ax[0].hist(results[i][0], histtype = 'step', color = cmap(i),
                    label = 'M_{DM} = ' + '{:.2e}'.format(vali * material.m) + ' eV')
            ax[1].hist(results[i][1], histtype = 'step', color = cmap(i))
        else:
            ax[0].hist(results[i][0], histtype = 'step', color = cmap(i))
            ax[1].hist(results[i][1], histtype = 'step', color = cmap(i),
                    label = 'M_{DM} = ' + '{:.2e}'.format(vali * material.m) + ' eV')
        
    ax[0].legend()
    ax[0].set_title('Mediator Mass = ' + str(mediator_mass))
    
    ax[0].set_xlabel('Energy [$\Delta$]')
    ax[0].yaxis.set_ticks_position('both')
    ax[0].yaxis.tick_right()
    ax[0].text(1 + 0.8e-5, 65, 'QP')
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
     
    ax[1].set_xlabel('Energy [$\Delta$]')
    ax[1].yaxis.set_ticks_position('both')
    ax[1].yaxis.tick_right()
    ax[1].text(0.17, 700, 'PH')
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    
    plt.savefig('../graph/energy_leaf_QP+PH_' + str(mat_name) + '_' + 
                "{:.2e}".format(np.min(m_nt * material.m)) + "-{:.2e}".format(np.max(m_nt * material.m)) +
                '_MedMass_' + "{:.2e}".format(mediator_mass) + '.pdf')


def make_graph_dep_energy(m_nt, results, mediator_mass, mat_name):
    
    cmap = get_cmap('viridis', len(m_nt))
    
    
    for i, vali in enumerate(m_nt):
        plt.hist(results[i][2], histtype = 'step', color = cmap(i),
                    label = 'M_{DM} = ' + '{:.2e}'.format(vali * material.m) + ' eV')
        
    plt.legend()
    plt.title('Mediator mass = {:.2e}'.format(mediator_mass))
    
    plt.xlabel('Energy [$\Delta$]')
    plt.yscale('log')
    plt.xscale('log')

    
    plt.savefig('../graph/dep_energy_' + str(mat_name) + '_' + 
                "{:.2e}".format(np.min(m_nt * material.m)) + "-{:.2e}".format(np.max(m_nt * material.m)) +
                '_MedMass_' + str(mediator_mass) + '_density.pdf')


#}}}
#}}}

# Configuration
#{{{
KMS = 3.33564e-6  # km/s in natural units

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
m_nt     = [1e6, 1e7, 1e8, 1e9] / material.m #np.concatenate((
           #np.linspace(1, 9, 3) * 1e4, 
           #np.linspace(1, 9, 3) * 1e5
           #)) / material.m # Dark matter masses
N_events = np.array( [100] ) # Numero de eventos observados
#}}}

mediator_mass = [0, 1e3, 1e5]
with h5py.File('SILICON.h5','a') as data:

    for j, valj in enumerate(mediator_mass):
        mat_element = FiducialMatrixElement(mediator_mass = valj)
        try:
            gr = data.create_group("{:.2e}".format(valj))
        except:
            gr = data["{:.2e}".format(valj)]

# Let's check and eliminate from m_nt the masses that were already computed before
        already_done = [float(i) for i in list(gr.keys())]
        m_nt = m_nt[~np.in1d(m_nt * material.m, already_done)]
# --------------------------------------------------------------------------------
        if len(m_nt) > 0:
            print('\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\')
            print('\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\')
            print('Starting analysis with mediator mass {:.2e}'.format(valj))
            print('\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\')
            print('\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\')
            results = Parallel(n_jobs=5)(delayed(analyse)(i, matrix_element = mat_element) for i in tqdm(range(len(m_nt))))
            make_graph(m_nt, results, valj, 'SI')
            make_graph_dep_energy(m_nt, results, valj, 'SI')

            for i, vali in enumerate(m_nt):
                try:
                    gr1 = gr.create_group("{:.2e}".format(vali * material.m))
                except:
                    gr1 = gr["{:.2e}".format(vali * material.m)]
                try:
                    gr1.create_dataset('QP_e', data = results[i][0])
                    gr1.create_dataset('PH_e', data = results[i][1])
                    gr1.create_dataset('Dep_e', data = results[i][2])
                    gr1.create_dataset('QP_cos_theta', data = results[i][3])
                    gr1.create_dataset('PH_cos_theta', data = results[i][4])
                except:
                    print('Ya existe')
                    pass
        else:
            print('Nothing new to compute!')


