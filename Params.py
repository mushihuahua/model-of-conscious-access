from dataclasses import dataclass


@dataclass
class Params:
    tau_NMDA: int = 60  # E synaptic time constants (ms)
    tau_AMPA: int = 2   # E synaptic time constants (ms)
    g_NMDA: int = 1000  # Channel conductances (ms)
    g_AMPA: int = 10_000 # Channell conductances (ms)
    tau_GABA: int = 5   # I synaptic constant (ms)
    tau_r: int = 2      # Firing rate time constant (ms)  
    gamma_NMDA: float = 1.282  # synaptic rise constants
    gamma_AMPA: int = 2      # synaptic rise constants
    gamma_GABA: int = 2         # synaptic rise constants
    k_sup: float = 0.0         # NMDA fraction
    k_dp: float = 0.8          # NMDA fraction
    k_local: float = 0.91      # NMDA fraction
    p_sup: float = 1.0           # Long-range E/I targets
    p_dp: float = 0.015        # Long-range E/I targets
    z_min: float = 0.6         # Minimum exctiation values
    z_min_I: float = 0.218     # Minimum excitation values
    sigma_noise: float = 2.5   # std. dev. noise (pA)
    I_bg_E: float = 329.4      # Background inputs (pA)
    I_bg_I: int = 260        # Background inputs (pA)
    a: float = 0.135           # f-I curve (E cells) (Hz/pA)
    b: float = 54             # f-I curve (E cells) (Hz)
    d: float= 0.308           # f-I curve (E cells) (s)
    beta_i: float = 153.75        #f-I curve (I cells) (Hz/nA)
    I_th: int = 252           # f-I curve (I cells) (s)
    b1: float = 0.3            # Rescale FLN
    G_n_loc_E_E = 480   # Excitatory strengths NMDA (pA)
    G_a_loc_E_E = 4800  # Excitatory strengths AMPA (pA)
    G_n_loc_I_E = 10    # Excitatory strangths NMDA inhibition (pA)
    G_E_I = -8800       # Inhibitory strengths (pA)
    G_I_I = -120        # Inhibitory strengths (pA)
    G_E_NMDA = 1500      # Long range strength NMDA (pA)
    G_I_NMDA = 10.5      # Long range strength AMPA (pA)
    G_0 = 215            # Local balanced coupling (pA)
    I_stim = 250        # Stimulus strength (pA)

    # https://elifesciences.org/articles/72136

    chi_raw_min: int = 643
    chi_raw_max: int = 8337
