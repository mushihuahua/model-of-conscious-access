import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import math

from ParamsLocal import Params
from brian2 import ms, second, amp, pA, nA, Hz

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

params = Params()
simulation_time = 100
dt = 0.001

t_span = (0, simulation_time)
t_span_euler = np.arange(0, simulation_time, dt)
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
        self.chi_raw = 643 # spine count of area 9/46d

        self.sNDMA = np.zeros(len(t_span_euler))
        self.sAMPA = np.zeros(len(t_span_euler))
        self.sGABA = np.zeros(len(t_span_euler))

        self.noise = np.zeros(len(t_span_euler))

        self.rE = np.zeros(len(t_span_euler))
        self.rI = np.zeros(len(t_span_euler))

    def ode_model(self, t, v):
        s_NDMA, s_AMPA, s_GABA, rE, rI, I_noise = v

        stimulus = 0
        if(t <= 50):
            stimulus = params.I_stim

        dsNDMAdt = self.synaptic_dynamics(s_NDMA, rE, params.tau_NMDA, params.gamma_NMDA)
        dsAMPAdt = self.synaptic_dynamics(s_AMPA, rE, params.tau_AMPA, params.gamma_AMPA)
        dsGABAdt = self.synaptic_dynamics(s_GABA, rI, params.tau_GABA, params.gamma_GABA, gaba=True)

        dnoise_dt = self.ornstein_uhlenbeck_process(I_noise)

        I_total_E = self._excitatory_ndma_current(s_NDMA) 
        # + self._excitatory_ampa_current(s_AMPA) 
        + self._excitatory_gaba_current(s_GABA) 
        # + self._inhibitory_ndma_current(s_NDMA) 
        + I_noise + params.I_bg_E + stimulus

        I_total_I =  self._inhibitory_ndma_current(s_NDMA) + I_noise + params.I_bg_I 

        drEdt = self.rate_dynamics(rE, I_total_E, "E")
        drIdt = self.rate_dynamics(rI, I_total_I, "I")

        return [dsNDMAdt, dsAMPAdt, dsGABAdt, drEdt, drIdt, dnoise_dt]

    def euler_ode_model(self, v):

        self.sNDMA[0], self.sAMPA[0], self.sGABA[0], self.rE[0], self.rI[0], self.noise[0] = v

        for i in range(1, len(t_span_euler)):

            s_NDMA, s_AMPA, s_GABA, rE, rI, I_noise = self.sNDMA[i-1], self.sAMPA[i-1], self.sGABA[i-1], self.rE[i-1]*Hz, self.rI[i-1]*Hz, self.noise[i-1]*amp

            t = i*dt

            stimulus = 0
            if(t <= 50):
                stimulus = params.I_stim
            
            self.sNDMA[i] = s_NDMA + self.synaptic_dynamics(s_NDMA, rE, params.tau_NMDA, params.gamma_NMDA) * dt*second
            self.sAMPA[i] = s_AMPA + self.synaptic_dynamics(s_AMPA, rE, params.tau_AMPA, params.gamma_AMPA) * dt*second
            self.sGABA[i] = s_GABA + self.synaptic_dynamics(s_GABA, rI, params.tau_GABA, params.gamma_GABA, gaba=True) * dt*second

            self.noise[i] = I_noise + self.ornstein_uhlenbeck_process(I_noise) * dt*second

            I_total_E = self._excitatory_ndma_current(s_NDMA) 
            + self._excitatory_gaba_current(s_GABA) 
            + I_noise + params.I_bg_E + stimulus

            I_total_I =  self._inhibitory_ndma_current(s_NDMA) + I_noise + params.I_bg_I 

            self.rE[i] = rE + self.rate_dynamics(rE, I_total_E, "E") * dt*second
            self.rI[i] = rI + self.rate_dynamics(rI, I_total_I, "I") * dt*second

            print(t)

    def run(self):

        init_conditions = np.array((0.1, 0.1, 0.1,
                                    0.1, 0.1, 0))
        

        # Solve the ODE system 
        # result = solve_ivp(self.ode_model, t_span, init_conditions,
        #                    t_eval=t_eval, method='RK45')
        
        self.euler_ode_model(init_conditions)
        # return result
        

    def synaptic_dynamics(self, s, r, tau, gamma, gaba=False):
        if(gaba):
            dsdt = (-s / tau) + gamma * r
        else:
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
    
    def ornstein_uhlenbeck_process(self, I):
        dIdt = (-I + np.random.normal(0, 1) * np.sqrt(2 * params.sigma_noise**2)) / params.tau_AMPA
        return dIdt
    
def get_nullclines(model):
    res = 100

    sNDMA_vals = np.linspace(0, 1, res)
    sGABA_vals = np.linspace(0, 1, res)
    sNDMA_grid, sGABA_grid = np.meshgrid(sNDMA_vals, sGABA_vals)

    nullclineE = np.zeros_like(sNDMA_grid)
    nullclineI = np.zeros_like(sGABA_grid)

    I_noise = 0
    
    def rateFuncE(sE, sI):
        I_total_E = (
            model._excitatory_ndma_current(sE)
            + model._excitatory_gaba_current(sI)
            + I_noise + params.I_bg_E + params.I_stim
            )
        
        rE = model._threshold_function("E", I_total_E)

        return rE

    nullclineE = np.vectorize(lambda sE, sI : model.synaptic_dynamics(sE, rateFuncE(sE, sI), params.tau_NMDA, params.gamma_NMDA)/Hz + 0.3)(sNDMA_grid, sGABA_grid) 

    def rateFuncI(sE, sI):
        I_total_I = model._inhibitory_ndma_current(sE) + I_noise + params.I_bg_I
        
        rI = model._threshold_function("I", I_total_I)

        return rI

    nullclineI = np.vectorize(lambda sE, sI : model.synaptic_dynamics(sI, rateFuncI(sE, sI), params.tau_GABA, params.gamma_GABA))(sNDMA_grid, sGABA_grid) 


    
    plt.figure(figsize=(8, 6))
    CS1 = plt.contour(sNDMA_grid, sGABA_grid, nullclineE, levels=[0], colors='blue', label='dE/dt=0')
    CS2 = plt.contour(sNDMA_grid, sGABA_grid, nullclineI, levels=[0], colors='red', label='dI/dt=0')
    plt.xlabel('sE')
    plt.ylabel('sI')
    plt.ylim([-0.05, 1.05])
    plt.xlim([-0.05, 1.05])
    plt.title('Phase Potrait')
    plt.grid(True)
    plt.show()
    
    return nullclineE

if( __name__ == "__main__"):

    model = LocalModel()
    # result = model.run()

    nullcline = get_nullclines(model)

    print(nullcline)

