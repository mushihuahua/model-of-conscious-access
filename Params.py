from dataclasses import dataclass, field
from brian2 import ms, second, pA, nA, Hz
from brian2.units.fundamentalunits import Quantity


@dataclass
class Params:
    tau_NMDA: Quantity = field(default_factory=lambda: 60*ms)  # E synaptic time constants (ms)
    tau_AMPA: Quantity = field(default_factory=lambda: 2*ms)   # E synaptic time constants (ms)
    g_NMDA: Quantity = field(default_factory=lambda: 1000*pA)  # Channel conductances (pA)
    g_AMPA: Quantity = field(default_factory=lambda: 10_000*pA) # Channell conductances (pA)
    tau_GABA: Quantity = field(default_factory=lambda: 5*ms)   # I synaptic constant (ms)
    tau_r: Quantity = field(default_factory=lambda: 2*ms)      # Firing rate time constant (ms)  
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
    sigma_noise: Quantity = field(default_factory=lambda: 2.5*pA)   # std. dev. noise (pA)
    I_bg_E: Quantity = field(default_factory=lambda: 329.4*pA)      # Background inputs (pA)
    I_bg_I: Quantity = field(default_factory=lambda: 260*pA)        # Background inputs (pA)
    a: Quantity = field(default_factory=lambda: 0.135*(Hz/pA))           # f-I curve (E cells) (Hz/pA)
    b: Quantity = field(default_factory=lambda: 54*Hz)             # f-I curve (E cells) (Hz)
    d: Quantity = field(default_factory=lambda: 0.308*second)           # f-I curve (E cells) (s)
    beta_i: Quantity = field(default_factory=lambda: 153.75*(Hz/nA))        #f-I curve (I cells) (Hz/nA)
    I_th: Quantity = field(default_factory=lambda: 252*pA)          # f-I curve (I cells) (Hz)
    b1: float = 0.3            # Rescale FLN
    G_n_loc_E_E: Quantity = field(default_factory=lambda: 480*pA)   # Excitatory strengths NMDA (pA)
    G_a_loc_E_E: Quantity = field(default_factory=lambda: 4800*pA)  # Excitatory strengths AMPA (pA)
    G_n_loc_I_E: Quantity = field(default_factory=lambda: 10*pA)    # Excitatory strangths NMDA inhibition (pA)
    G_E_I: Quantity = field(default_factory=lambda: -8800*pA)       # Inhibitory strengths (pA)
    G_I_I: Quantity = field(default_factory=lambda: -120*pA)        # Inhibitory strengths (pA)
    G_E_NMDA: Quantity = field(default_factory=lambda: 1500*pA)      # Long range strength NMDA (pA)
    G_I_NMDA: Quantity = field(default_factory=lambda: 10.5*pA)      # Long range strength AMPA (pA)
    G_E_AMPA: Quantity = field(default_factory=lambda: 3500*pA)      # Long range strength NMDA (pA)
    G_I_AMPA: Quantity = field(default_factory=lambda: 105*pA)      # Long range strength AMPA (pA)
    G_0: Quantity = field(default_factory=lambda: 215*pA)            # Local balanced coupling (pA)
    I_stim: Quantity = field(default_factory=lambda: 325*pA)       # Stimulus strength (pA)
    I_vig: Quantity = field(default_factory=lambda: 95*pA)       # Vigilance strength (pA)

    simulation_time: float = 1.5 # 1 second simulation
    dt: float = 0.0001

    stimulus_start: float = 0.50
    stimulus_end: float = 0.55

    distractor_start: float = 1
    distractor_end: float = -1.05

    num_of_trials: int = 1

    # https://elifesciences.org/articles/72136

    chi_raw_min: int = 643
    chi_raw_max: int = 8337

    num_of_source_areas: int = 40
    num_of_target_areas: int = 40