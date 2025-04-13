import numpy as np
from scipy.integrate import solve_ivp
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd
import math

from Params import Params
from utils.spines import spine_count_gradient_long_range, normalised_spine_count_long_range
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

class LongRangeModel:
    """The long range model"""

    def __init__(self):
        # Store parameters
        self.params = params
        self.k = 0
        # self.chi_raw = 7800 # spine count of area 9/46d from literature

        self.sNDMA = np.zeros((params.num_of_target_areas, len(t_span_euler)))
        self.sAMPA = np.zeros((params.num_of_target_areas, len(t_span_euler)))
        self.sGABA = np.zeros((params.num_of_target_areas, len(t_span_euler)))

        self.noise = np.zeros((params.num_of_target_areas, len(t_span_euler)))

        self.rE = np.zeros((params.num_of_target_areas, len(t_span_euler)))
        self.rI = np.zeros((params.num_of_target_areas, len(t_span_euler)))

        self.sln = self.fln = np.zeros((params.num_of_target_areas, params.num_of_source_areas))

        self.W = np.zeros((params.num_of_target_areas, params.num_of_source_areas))

    def _connection_strength_w(self, k, l):
        return (self.fln[k, l]**params.b1)/(np.sum(self.fln, axis=1)[k]**params.b1)

    def init_anatomy(self):
        subgraph_data = sio.loadmat('data/labeled_neurons.mat')
        self.sln = subgraph_data['HierOrderedSLNsubgraph']
        self.fln = subgraph_data['HierOrderedFLNsubgraph']

        subgraph_data = sio.loadmat('data/spine_counts.mat')
        self.spine_counts = subgraph_data['spine_count_lyon_regions_40'].ravel()

        K, L = np.meshgrid(range(params.num_of_target_areas), range(params.num_of_source_areas), indexing='ij')
        self.W = np.vectorize(self._connection_strength_w)(K, L)

    # def ode_model(self, t, v):
    #     s_NDMA, s_AMPA, s_GABA, rE, rI, I_noise = v

    #     stimulus = 0
    #     if(t <= 50):
    #         stimulus = params.I_stim

    #     dsNDMAdt = self.synaptic_dynamics(s_NDMA, rE, params.tau_NMDA, params.gamma_NMDA)
    #     dsAMPAdt = self.synaptic_dynamics(s_AMPA, rE, params.tau_AMPA, params.gamma_AMPA)
    #     dsGABAdt = self.synaptic_dynamics(s_GABA, rI, params.tau_GABA, params.gamma_GABA, gaba=True)

    #     dnoise_dt = self.ornstein_uhlenbeck_process(I_noise)

    #     I_total_E = self._excitatory_ndma_current(s_NDMA) 
    #     # + self._excitatory_ampa_current(s_AMPA) 
    #     + self._excitatory_gaba_current(s_GABA) 
    #     # + self._inhibitory_ndma_current(s_NDMA) 
    #     + I_noise + params.I_bg_E + stimulus

    #     I_total_I =  self._inhibitory_ndma_current(s_NDMA) + I_noise + params.I_bg_I 

    #     drEdt = self.rate_dynamics(rE, I_total_E, "E")
    #     drIdt = self.rate_dynamics(rI, I_total_I, "I")

    #     return [dsNDMAdt, dsAMPAdt, dsGABAdt, drEdt, drIdt, dnoise_dt]

    def euler_ode_model(self, v):
        
        self.sNDMA[:,0], self.sAMPA[:,0], self.sGABA[:,0], self.rE[:,0], self.rI[:,0], self.noise[:,0] = v

        for i in range(1, len(t_span_euler)):

            s_NDMA, s_AMPA, s_GABA, rE, rI, I_noise = self.sNDMA[:, i-1], self.sAMPA[:, i-1], self.sGABA[:, i-1], self.rE[:, i-1]*Hz, self.rI[:, i-1]*Hz, self.noise[:, i-1]*amp

            t = i*dt

            stimulus = 0
            if(t <= 50):
                stimulus = params.I_stim
            
            self.sNDMA[:, i] = s_NDMA + self.synaptic_dynamics(s_NDMA, rE, params.tau_NMDA, params.gamma_NMDA) * dt*second
            self.sAMPA[:, i] = s_AMPA + self.synaptic_dynamics(s_AMPA, rE, params.tau_AMPA, params.gamma_AMPA) * dt*second
            self.sGABA[:, i] = s_GABA + self.synaptic_dynamics(s_GABA, rI, params.tau_GABA, params.gamma_GABA, gaba=True) * dt*second

            self.noise[:, i] = I_noise + self.ornstein_uhlenbeck_process(I_noise) * dt*second

            # print(self._excitatory_ndma_current_long_range(s_NDMA))
            # print(np.array(list(map(self._dendritic_function_F, self._excitatory_ndma_current_long_range(s_NDMA))))* amp) 

            I_total_E = (
                np.array(list(map(self._dendritic_function_F, self._excitatory_ndma_current_long_range(s_NDMA)))) * amp
                + np.array(list(map(self._dendritic_function_F, self._excitatory_ampa_current_long_range(s_AMPA)))) * amp
                + self._excitatory_ndma_current(s_NDMA) 
                + self._excitatory_gaba_current(s_GABA) 
                + I_noise + params.I_bg_E + stimulus
            )

            # print(rE)

            I_total_I = (
                self._inhibitory_ndma_current_long_range(s_NDMA)
                + self._inhibitory_ampa_current_long_range(s_AMPA)
                + self._inhibitory_ndma_current(s_NDMA) 
                + I_noise + params.I_bg_I 
            )

            self.rE[:, i] = rE + self.rate_dynamics(rE, I_total_E, "E") * dt*second
            self.rI[:, i] = rI + self.rate_dynamics(rI, I_total_I, "I") * dt*second

            print(t)

    def run(self):

        self.init_anatomy()

        init_conditions = np.array((0.1, 0.1, 0.1,
                                    0.1, 0.1, 0))
        

        # Solve the ODE system 
        # result = solve_ivp(self.ode_model, t_span, init_conditions,
        #                    t_eval=t_eval, method='RK45')
        
        self.euler_ode_model(init_conditions)
        # print(self.W)
        

    def synaptic_dynamics(self, s, r, tau, gamma, gaba=False):
        if(gaba):
            dsdt = (-s / tau) + gamma * r
        else:
            dsdt = (-s / tau) + (1 - s) * gamma * r
        return dsdt
    
    def rate_dynamics(self, r: float, I_total: int, population: str):
        drdt = (-r + self._threshold_function(population, I_total)) / params.tau_r
        return drdt
    
    def ornstein_uhlenbeck_process(self, I):
        dIdt = (-I + np.random.normal(0, 1) * np.sqrt(2 * params.sigma_noise**2)) / params.tau_AMPA
        return dIdt

    def _threshold_function(self, population: str, I_total: int):

        if(population == "E"):
            val = (params.a * I_total - params.b)
            return val / (1 - np.exp(-params.d * val))

        if(population == "I"):
            return params.beta_i * (I_total - params.I_th) * (I_total >= params.I_th)
        
        logger.error("Crashing... `threshold_function` got bad input for `population` parameter")
        exit()

    def _excitatory_ndma_current(self, S):
        I = spine_count_gradient_long_range(self.spine_counts, "E") * params.G_n_loc_E_E * S
        return I
    
    def _excitatory_ampa_current(self, S):
        I = spine_count_gradient_long_range(self.spine_counts, "E") * (1 - params.k_local) * params.G_a_loc_E_E * S
        return I
    
    def _excitatory_gaba_current(self, S):
        I = params.G_E_I * S
        return I
    
    def _inhibitory_ndma_current(self, S):
        I = spine_count_gradient_long_range(self.spine_counts, "I") * params.G_n_loc_I_E * S
        return I
    
    def _excitatory_ndma_current_long_range(self, S):
        sum = 0
        for l in range(params.num_of_source_areas):
            sum += self.W[:, l] * (self.sln[:, l]*params.k_sup*params.p_sup +
                                   (1 - self.sln[:, l])*params.k_dp*params.p_dp) * S[l]
                                   
        I = params.G_E_NMDA * spine_count_gradient_long_range(self.spine_counts, "E") * sum
        return I

    def _excitatory_ampa_current_long_range(self, S):
        sum = 0
        for l in range(params.num_of_source_areas):
            sum += self.W[:, l] * (self.sln[:, l]*(1-params.k_sup)*params.p_sup +
                                   (1 - self.sln[:, l])*(1-params.k_dp)*params.p_dp) * S[l]
                                   
        I = params.G_E_AMPA * spine_count_gradient_long_range(self.spine_counts, "E") * sum
        return I

    def _dendritic_function_F(self, X):
        
        if(X <= 0*pA):
            return 0*pA

        if(X >= 300*pA):
            return 300*pA

        return X

    def _inhibitory_ndma_current_long_range(self, S):
        sum = 0
        for l in range(params.num_of_source_areas):
            sum += self.W[:, l] * (self.sln[:, l]*params.k_sup*(1-params.p_sup) +
                                   (1 - self.sln[:, l])*params.k_dp*(1-params.p_dp)) * S[l]
                                   
        I = params.G_I_NMDA * spine_count_gradient_long_range(self.spine_counts, "I") * sum
        return I

    def _inhibitory_ampa_current_long_range(self, S):
        sum = 0
        for l in range(params.num_of_source_areas):
            sum += self.W[:, l] * (self.sln[:, l]*(1-params.k_sup)*(1-params.p_sup) +
                                   (1 - self.sln[:, l])*(1-params.k_dp)*(1-params.p_dp)) * S[l]
                                   
        I = params.G_I_AMPA * spine_count_gradient_long_range(self.spine_counts, "I") * sum
        return I


    
if( __name__ == "__main__"):

    model = LongRangeModel()
    result = model.run()

    plt.plot(t_span_euler, model.rE[23], color="red")
    plt.plot(t_span_euler, model.rI[23], color="blue")

    # plt.plot(t_span_euler, model.sNDMA, label="NMDA")
    # plt.plot(t_span_euler, model.sGABA, label="GABA")
    # plt.plot(t_span_euler, model.sAMPA, label="AMPA")
    plt.legend()
    plt.show()
