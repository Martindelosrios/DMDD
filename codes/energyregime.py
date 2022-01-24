# Generates histograms for the energy deposits.

import numpy as np
import matplotlib.pyplot as plt

from scdc.ensemble import Ensemble
from scdc.event import Event
from scdc.particle import Quasiparticle
from scdc.materials import ALUMINUM, SILICON
from scdc.initial.distribution.integral import InitialSampler
from scdc.initial.halo import StandardHaloDistribution
from scdc.initial.response import HybridResponseFunction
from scdc.initial.matrix_element import FiducialMatrixElement

KMS = 3.33564e-6  # km/s in natural units
aluminum = ALUMINUM
vdf_SI = StandardHaloDistribution(
    v_0    = 220 * KMS / aluminum.v, 
    v_esc  = 550 * KMS / aluminum.v,
    v_wind = 230 * KMS / aluminum.v
)
vdf_SI_iso = StandardHaloDistribution(
    v_0    = 220 * KMS / aluminum.v, 
    v_esc  = 550 * KMS / aluminum.v,
    v_wind = 0 * KMS / aluminum.v
)
response_SI = HybridResponseFunction(aluminum, 1)

silicon = ALUMINUM
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
response_SI = HybridResponseFunction(silicon, 1)

m_nt_AL = [1e3] / aluminum.m
m_nt_SI_eV= [1e3,1e4,1e5]
m_nt_SI = m_nt_SI_eV / silicon.m

me_light = FiducialMatrixElement(mediator_mass = 0)
me_heavy_AL = FiducialMatrixElement(mediator_mass = 0)
me_heavy_SI = FiducialMatrixElement(mediator_mass = 10)

fig, ax = plt.subplots(1, 2,)

for i in np.arange(np.size(m_nt_SI)):
    # sampler_SI_iso_light = InitialSampler(m_nt_SI[i], me_light, silicon, response_SI, vdf_SI_iso, n_cq = 20, n_rq = 20)
    # sampler_SI_iso_heavy = InitialSampler(m_nt_SI[i], me_heavy_AL, silicon, response_SI, vdf_SI_iso, n_cq = 20, n_rq = 20)
    sampler_SI_light = InitialSampler(m_nt_SI[i], me_light, silicon, response_SI, vdf_SI, n_cq = 20, n_rq = 20)
    sampler_SI_heavy = InitialSampler(m_nt_SI[i], me_heavy_SI, silicon, response_SI, vdf_SI, n_cq = 20, n_rq = 20)

    # simulation_SI_iso_light = sampler_SI_iso_light.ensemble(1000)
    # simulation_SI_iso_heavy = sampler_SI_iso_heavy.ensemble(1000)
    simulation_SI_light = sampler_SI_light.ensemble(1000)
    simulation_SI_heavy = sampler_SI_heavy.ensemble(1000)

    '''for e in simulation_SI_iso_light:
        for p in e.out:
            p.dest.final = True
    for e in simulation_SI_iso_heavy:
        for p in e.out:
            p.dest.final = True'''
    for e in simulation_SI_light:
        for p in e.out:
            p.dest.final = True
    for e in simulation_SI_heavy:
        for p in e.out:
            p.dest.final = True
    
    '''omega_SI_iso_light=[]
    for e in simulation_SI_iso_light:
        E=0
        for p in e.leaf_particles.quasiparticles:
            E=E+p.energy
        if E<np.inf:
            omega_SI_iso_light.append(E)

    omega_SI_iso_heavy=[]
    for e in simulation_SI_iso_heavy:
        E=0
        for p in e.leaf_particles.quasiparticles:
            E=E+p.energy
        if E<np.inf:
            omega_SI_iso_heavy.append(E)'''

    omega_SI_light=[]
    for e in simulation_SI_light:
        E=0
        for p in e.leaf_particles.quasiparticles:
            E=E+p.energy
        if E<np.inf:
            omega_SI_light.append(E)

    omega_SI_heavy=[]
    for e in simulation_SI_heavy:
        E=0
        for p in e.leaf_particles.quasiparticles:
            E=E+p.energy
        if E<np.inf:
            omega_SI_heavy.append(E)

    ax[0].hist(omega_SI_light, histtype='step', lw=3, density = True, label = 'm='+str(m_nt_SI_eV[i])+' eV')
    # ax[0].hist(omega_SI_iso_light, histtype='step', lw=1, density = True, label = 'Iso')
    ax[1].hist(omega_SI_heavy, histtype='step', lw=3, density = True, label = 'm='+str(m_nt_SI_eV[i])+' eV')
    # ax[1].hist(omega_SI_iso_heavy, histtype='step', lw=1, density = True, label = 'Iso')

ax[0].set_ylabel('Density')
ax[0].set_xlabel('E ($\Delta$)')
ax[0].set_title('Light Mediator (m=0)')
ax[0].legend()

ax[1].set_ylabel('Density')
ax[1].set_xlabel('E ($\Delta$)')
ax[1].set_title('Heavy Mediator (m=10)')
ax[1].legend()

plt.savefig('E_AL_proof.pdf')
plt.show()