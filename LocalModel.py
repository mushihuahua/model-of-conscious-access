import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
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
simulation_time = 100
dt = 0.01

t_span = (0, simulation_time)
t_eval = np.linspace(0, simulation_time, int(simulation_time / dt))

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

    def ode_model(self, t, v):
        s_NDMA, s_AMPA, s_GABA, rE, rI, I_noise = v

        dsNDMAdt = self.synaptic_dynamics(s_NDMA, rE, params.tau_NMDA, params.gamma_NMDA)
        dsAMPAdt = self.synaptic_dynamics(s_AMPA, rE, params.tau_AMPA, params.gamma_AMPA)
        dsGABAdt = self.synaptic_dynamics(s_GABA, rI, params.tau_GABA, params.gamma_GABBA, gaba=True)

        dnoise_dt = self.ornstein_uhlenbeck_process(I_noise)

        I_total_E = self._excitatory_ndma_current(s_NDMA) 
        + self._excitatory_ndma_current(s_AMPA) 
        + self._excitatory_ndma_current(s_GABA) 
        + self._inhibitory_ndma_current(s_NDMA) 
        + self.ornstein_uhlenbeck_process(I_noise)
        + params.I_bg_E 

        I_total_I =  self._inhibitory_ndma_current(s_NDMA) + self.ornstein_uhlenbeck_process(I_noise) + params.I_bg_E 

        drEdt = self.rate_dynamics(rE, I_total_E, "E")
        drIdt = self.rate_dynamics(rI, I_total_I, "I")

        return [dsNDMAdt, dsAMPAdt, dsGABAdt, drEdt, drIdt, dnoise_dt]

    def run(self):

        init_conditions = [0.1, 0.1, 0.1,
                           0.1, 0.1, 0.1]

        # Solve the ODE system 
        result = solve_ivp(self.ode_model, t_span, init_conditions,
                           t_eval=t_eval, method='RK45')
        
        return result
        

    def synaptic_dynamics(self, s, r, tau, gamma, gaba=False):
        dsdt = (-s / tau) + (not gaba * (1 - s)) * gamma * r
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
    
    def ornstein_uhlenbeck_process(self, I):
        dIdt = (-I + np.random.normal(0, 1) * math.sqrt(params.tau_AMPA * params.sigma_noise**2)) / params.tau_AMPA
        return dIdt
    
if( __name__ == "__main__"):

    model = LocalModel()
    result = model.run()

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(result.t, result.y[3], label='Excitatory')
    plt.plot(result.t, result.y[4], label='Inhibitory')
    plt.show()
