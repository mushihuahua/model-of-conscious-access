from dataclasses import dataclass
from brian2 import ms, second, pA, nA, Hz
from brian2.units.fundamentalunits import Quantity


@dataclass
class Params:
    tau_NMDA = 0.66*ms  # E synaptic time constants (ms)
    tau_AMPA = 2*ms   # E synaptic time constants (ms)
    g_NMDA = 1000*pA  # Channel conductances (pA)
    g_AMPA = 10_000*pA # Channell conductances (pA)
    tau_GABA = 5*ms   # I synaptic constant (ms)
    tau_r = 2*ms      # Firing rate time constant (ms)  
    gamma_NMDA: float = 2  # synaptic rise constants
    gamma_AMPA: int = 2      # synaptic rise constants
    gamma_GABA: int = 2         # synaptic rise constants
    k_sup: float = 0.0         # NMDA fraction
    k_dp: float = 0.8          # NMDA fraction
    k_local: float = 0.91      # NMDA fraction
    p_sup: float = 1.0           # Long-range E/I targets
    p_dp: float = 0.015        # Long-range E/I targets
    z_min: float = 0.6         # Minimum exctiation values
    z_min_I: float = 0.218     # Minimum excitation values
    sigma_noise = 2.5*pA   # std. dev. noise (pA)
    I_bg_E = 430*pA      # Background inputs (pA)
    I_bg_I = 760*pA        # Background inputs (pA)
    a = 0.27*(Hz/pA)           # f-I curve (E cells) (Hz/pA)
    b = 108*Hz             # f-I curve (E cells) (Hz)
    d = 0.01*second           # f-I curve (E cells) (s)
    beta_i = 153.75*(Hz/nA)        #f-I curve (I cells) (Hz/nA)
    I_th = 252*pA          # f-I curve (I cells) (Hz)
    b1: float = 0.3            # Rescale FLN
    G_n_loc_E_E = 17000*pA   # Excitatory strengths NMDA (pA)
    G_a_loc_E_E = 4800*pA  # Excitatory strengths AMPA (pA)
    G_n_loc_I_E = 400*pA    # Excitatory strangths NMDA inhibition (pA)
    G_E_I = -8800*pA       # Inhibitory strengths (pA)
    G_I_I = -120*pA        # Inhibitory strengths (pA)
    G_E_NMDA = 15000*pA      # Long range strength NMDA (pA)
    G_I_NMDA = 105*pA      # Long range strength AMPA (pA)
    G_0 = 215*pA            # Local balanced coupling (pA)
    I_stim = 250*pA       # Stimulus strength (pA)

    # https://elifesciences.org/articles/72136

    chi_raw_min: int = 643
    chi_raw_max: int = 8337
