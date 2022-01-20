# Neccesay libraries
# {{{
import pickle
import numpy as np
from scipy import stats
from tqdm import tqdm
import h5py

from scdc.ensemble import Ensemble
from scdc.event import Event
from scdc.particle import Quasiparticle
from scdc.materials import ALUMINUM, NIOBIUM, SILICON
#from scdc import plot  # Contains matplotlib configuration code
from scdc.initial.distribution.integral import InitialSampler
from scdc.initial.halo import StandardHaloDistribution
from scdc.initial.response import HybridResponseFunction
from scdc.initial.matrix_element import FiducialMatrixElement
# }}}

# Configuration Parameters
#{{{
name = '../data/test_bins' # Just a name to save the files
nsim = 10 # Number of simulations

KMS = 3.33564e-6  # km/s in natural units

silicon = SILICON # Detector Material

# Dark matter velocity distributions
vdf_SI = StandardHaloDistribution(
    v_0    = 220 * KMS / silicon.v, 
    v_esc  = 550 * KMS / silicon.v,
    v_wind = 230 * KMS / silicon.v
)
vdf_SI_iso = StandardHaloDistribution(
    v_0    = 220 * KMS / silicon.v, 
    v_esc  = 550 * KMS / silicon.v,
    v_wind = 0 * KMS / silicon.v
)

# Material response function
response_SI = HybridResponseFunction(silicon, 1) # The 1 is the coherence sign. Can be +1 or -1

# Scattering M2 matrix
me_light = FiducialMatrixElement(mediator_mass = 0)

# Dark matter masses
m_nt_SI = np.concatenate((np.linspace(1, 9, 9) * 1e6, np.linspace(1, 9, 9) * 1e7)) / silicon.m

# Number of events
N = np.array( [1500, 5000] )

# Number of bins in the detector
nbins = np.array( [2, 5, 10, 100, 1000] )
#}}}

# Loop
#{{{

# HDF5 where we will save the output
with h5py.File(name + '.hdf5', 'a') as f:

    for j in range(np.size(N)): # Loop over the number of events
        print('Starting even ' + str(j+1) + ' of ' + str(len(N)))
        
        # Let's create a group for each number of events
        if 'Nevents_' + str(N[j]) in f.keys():
            eve = f['Nevents_' + str(N[j])]
        else:
            eve = f.create_group('Nevents_' + str(N[j]))
    
        prev_sims = len(eve.keys()) # Just to check if there are previous simus and avoid writting above them.
        for k in range(nsim): # Loop over the number of simulations
            print('Starting simu ' + str(k+1) + ' of ' + str(nsim))
        
            # Let's create a group for each simulation
            sim = eve.create_group('sim_' + str(prev_sims + k))
    
            for i in tqdm(range(np.size(m_nt_SI))): # Loop over dark matter masses
              
                sampler_SI_iso_light    = InitialSampler(m_nt_SI[i], me_light, silicon, response_SI, vdf_SI_iso, n_cq = 20, n_rq = 20)
                sampler_SI_light        = InitialSampler(m_nt_SI[i], me_light, silicon, response_SI, vdf_SI, n_cq = 20, n_rq = 20)
                simulation_SI_iso_light = sampler_SI_iso_light.ensemble(N[j])
                simulation_SI_light     = sampler_SI_light.ensemble(N[j])
                
                simulation_SI_iso_light.chain()
                simulation_SI_light.chain()
                
                QP_set = sim.create_dataset('QP_dist_' + str(i), data = simulation_SI_light.leaf_particles.quasiparticles.cos_theta)
                Ph_set = sim.create_dataset('Ph_dist_' + str(i), data = simulation_SI_light.leaf_particles.phonons.cos_theta)
                QP_set.attrs['dm_m'] = m_nt_SI[i] * silicon.m 
                Ph_set.attrs['dm_m'] = m_nt_SI[i] * silicon.m 
                QP_set.attrs['pval'] = stats.ks_2samp(simulation_SI_iso_light.leaf_particles.quasiparticles.cos_theta, simulation_SI_light.leaf_particles.quasiparticles.cos_theta)[1]
                Ph_set.attrs['pval'] = stats.ks_2samp(simulation_SI_iso_light.leaf_particles.phonons.cos_theta, simulation_SI_light.leaf_particles.phonons.cos_theta)[1]
    
                for h in range(len(nbins)): # Loop over number of detector angular bins
                    qp_iso, bin_edges = np.histogram(simulation_SI_iso_light.leaf_particles.quasiparticles.cos_theta, bins = nbins[h])
                    bin_widths  = bin_edges[1] - bin_edges[0]
                    bin_centers = bin_edges[1:] - bin_widths/2  

                    ph_iso, _ = np.histogram(simulation_SI_iso_light.leaf_particles.phonons.cos_theta, bins = bin_edges)
                    qp, _     = np.histogram(simulation_SI_light.leaf_particles.quasiparticles.cos_theta, bins = bin_edges)
                    ph, _     = np.histogram(simulation_SI_light.leaf_particles.phonons.cos_theta, bins = bin_edges)

                    for ii in range(len(bin_centers)):
                      if ii == 0:
                        qp_obs_iso = np.asarray([bin_centers[ii]] * qp_iso[ii])
                        qp_obs     = np.asarray([bin_centers[ii]] * qp[ii])
                        ph_obs_iso = np.asarray([bin_centers[ii]] * ph_iso[ii])
                        ph_obs     = np.asarray([bin_centers[ii]] * ph[ii])
                      else:
                        qp_obs_iso = np.concatenate(( qp_obs_iso, np.asarray([bin_centers[ii]] * qp_iso[ii]) ))
                        qp_obs     = np.concatenate(( qp_obs, np.asarray([bin_centers[ii]] * qp[ii]) ))
                        ph_obs_iso = np.concatenate(( ph_obs_iso, np.asarray([bin_centers[ii]] * ph_iso[ii]) ))
                        ph_obs     = np.concatenate(( ph_obs, np.asarray([bin_centers[ii]] * ph[ii]) ))
                    
                    
                    QP_set.attrs['pval_' + str(nbins[h])] = stats.ks_2samp(qp_obs_iso, qp_obs)[1]
                    Ph_set.attrs['pval_' + str(nbins[h])] = stats.ks_2samp(ph_obs_iso, ph_obs)[1]
      
#}}}

