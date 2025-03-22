import numpy as np
from scipy.integrate import solve_ivp
import math

from Params import Params

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

params = Params()

def normalised_spine_count(k: int, chi_raw: int):
    """Calculate and return the normalised spine count for area `k`
    
       Inputs:
        `k`: Cortical area.
        `χ_raw`: The spine count for area `k`.
    """

    chi_normalised = (chi_raw - params.chi_raw_min) / (params.chi_raw_max - params.chi_raw_min)
    return chi_normalised

def spine_count_gradient(k: int, chi_raw: int, population: str):
    """Calculate and return the spine count modulation variable for area `k`"""

    z_min = params.z_min if population == "E" else params.z_min_I

    z_k = z_min + normalised_spine_count(k, chi_raw) * (1 - z_min)
    return z_k


class LocalModel:

    """The local cortical area model"""
    def __init__(self):
        # Store parameters
        self.params = params
        self.k = 0
        self.chi_raw = 7800 # spine count of area 9/46d

    def ode_model(self):
        pass

    def run(self):
        pass

    def synaptic_dynamics(self, s, r, tau, gamma):
        dsdt = (-s / tau) + (1 - s) * gamma * r
        return dsdt
    
    def rate_dynamics(self, r: float, I_total: int, population: str):
        drdt = (-r + self._threshold_function(population, I_total)) / params.tau_r
        return drdt

    def _threshold_function(self, population: str, I_total: int):

        if(population == "E"):
            val = (params.a * I_total - params.b)
            return val / (1 - math.exp(-params.d * val))

        if(population == "I"):
            return params.beta_i * (I_total - params.I_th) * (I_total >= params.I_th)
        
        logger.error("Crashing... `threshold_function` got bad input for `population` parameter")
        exit()

    def _excitatory_ndma_current(self, s):
        I = spine_count_gradient(self.k, self.chi_raw, "E") * params.k_local * params.G_n_loc_E_E * s
        return I
    
    def _excitatory_ampa_current(self, s):
        I = spine_count_gradient(self.k, self.chi_raw, "E") * (1 - params.k_local) * params.G_a_loc_E_E * s
        return I
    
    def _excitatory_gaba_current(self, s):
        I = params.G_E_I * s
        return I
    
    def _inhibitory_ndma_current(self, s):
        I = spine_count_gradient(self.k, self.chi_raw, "I") * params.G_n_loc_I_E * s
        return I
    
    def ornstein_uhlenbeck_process(I):
        dIdt = -I + np.random.normal(0, 1) * math.sqrt(params.tau_AMPA * params.sigma_noise**2)
        return dIdt
    
if( __name__ == "__main__"):

    model = LocalModel()
    model.run()