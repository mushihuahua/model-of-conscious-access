import numpy as np
from scipy.integrate import solve_ivp
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd
import math

from Params import Params
from utils.spines import spine_count_gradient_long_range, normalised_spine_count_long_range
from utils.cache import Cache
from brian2 import ms, second, amp, pA, nA, Hz

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

class LongRangeModel:
    """The long range model"""

    def __init__(self, params):
        # Store parameters
        self.params = params
        self.cache = Cache()

        self.simulation_time = params.simulation_time # 1 second simulation
        self.dt = params.dt

        self.stimulus_start = params.stimulus_start
        self.stimulus_end = params.stimulus_end

        self.distractor_start = params.distractor_start
        self.distractor_end = params.distractor_end

        self.num_of_trials = 1

        self.t_span = (0, self.simulation_time)
        self.t_span_euler = np.arange(0, self.simulation_time, self.dt)
        self.t_eval = np.linspace(0, self.simulation_time, int(self.simulation_time / self.dt))

        # self.chi_raw = 7800 # spine count of area 9/46d from literature
        self._init_anatomy()

    def _connection_strength_w(self, k, l):
        return (self.fln[k, l]**self.params.b1)/(np.sum(self.fln, axis=1)[k]**self.params.b1)

    def _init_variables(self): 
        self.sNDMA = np.zeros((self.params.num_of_target_areas, len(self.t_span_euler), 2))
        self.sAMPA = np.zeros((self.params.num_of_target_areas, len(self.t_span_euler), 2))
        self.sGABA = np.zeros((self.params.num_of_target_areas, len(self.t_span_euler)))

        self.noise = np.zeros((self.params.num_of_target_areas, len(self.t_span_euler), 3))

        self.rE = np.zeros((self.params.num_of_target_areas, len(self.t_span_euler), 2))
        self.rI = np.zeros((self.params.num_of_target_areas, len(self.t_span_euler)))

    def _init_anatomy(self):
        subgraph_data = sio.loadmat('data/labeled_neurons.mat')
        self.sln = subgraph_data['HierOrderedSLNsubgraph']
        self.fln = subgraph_data['HierOrderedFLNsubgraph']

        subgraph_data = sio.loadmat('data/spine_counts.mat')    
        self.spine_counts = subgraph_data['spine_count_lyon_regions_40'].ravel()
        area_names = subgraph_data['subgraph_hierarchical_order'].ravel()
        print(self.spine_counts)
        self.area_names = list(map(lambda x: str(x[0]), area_names))
        print(self.area_names)

        K, L = np.meshgrid(range(self.params.num_of_target_areas), range(self.params.num_of_source_areas), indexing="ij")
        self.W = np.vectorize(self._connection_strength_w)(K, L)
        # self.W[0,:] = 0
        # # print(self.W)

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
        
        self.sNDMA[:,0,:], self.sAMPA[:,0,:], self.sGABA[:,0], self.rE[:,0,:], self.rI[:,0], self.noise[:,0,:] = v

        for i in range(1, len(self.t_span_euler)):

            s_GABA, rI = self.sGABA[:,i-1], self.rI[:,i-1]*Hz

            t = i*self.dt

            vigilance_current = self.params.I_vig

            stimulus = 0
            if(t <= self.stimulus_end and t >= self.stimulus_start):
                stimulus = self.params.I_stim

            distractor = 0
            if(t <= self.distractor_end and t >= self.distractor_start):
                distractor = self.params.I_stim
            
            self.sGABA[:, i] = s_GABA + self.synaptic_dynamics(s_GABA, rI, self.params.tau_GABA, self.params.gamma_GABA, gaba=True) * self.dt*second

            for j in range(len(["E1", "E2"])):

                s_NDMA, s_AMPA = self.sNDMA[:, i-1, j], self.sAMPA[:, i-1, j]
                rE, I_noise = self.rE[:, i-1, j]*Hz, self.noise[:, i-1, j]*amp

                # print(self._excitatory_ampa_current_long_range(s_AMPA))
                # print(np.array(list(map(self._dendritic_function_F, self._excitatory_ampa_current_long_range(s_AMPA))))* amp) 

                self.sNDMA[:, i, j] = s_NDMA + self.synaptic_dynamics(s_NDMA, rE, self.params.tau_NMDA, self.params.gamma_NMDA) * self.dt*second
                self.sAMPA[:, i, j] = s_AMPA + self.synaptic_dynamics(s_AMPA, rE, self.params.tau_AMPA, self.params.gamma_AMPA) * self.dt*second

                self.noise[:, i, j] = I_noise + self.ornstein_uhlenbeck_process(I_noise) * self.dt*second

                k_local = s_NDMA/(s_NDMA+s_AMPA)

                total_current = (
                    np.array(list(map(self._dendritic_function_F, self._excitatory_ndma_current_long_range(s_NDMA)))) * amp
                    + np.array(list(map(self._dendritic_function_F, self._excitatory_ampa_current_long_range(s_AMPA)))) * amp
                    + self._excitatory_ndma_current(s_NDMA, k_local) 
                    + self._excitatory_ampa_current(s_AMPA, k_local)
                    + self._excitatory_gaba_current(s_GABA) 
                    + I_noise + self.params.I_bg_E
                )

                I_total_E = np.zeros(self.params.num_of_target_areas)*amp

                if(j == 0):
                    I_total_E[0] = total_current[0] + stimulus 
                    I_total_E[-int(0.75*len(I_total_E)):] = total_current[-int(0.75*len(I_total_E)):] + vigilance_current
                
                if(j == 1):
                    I_total_E[0] = total_current[0] + distractor 
                    I_total_E[-int(0.75*len(I_total_E)):] = total_current[-int(0.75*len(I_total_E)):] + vigilance_current
            
                I_total_E[1:-int(0.75*len(I_total_E))] = total_current[1:-int(0.75*len(I_total_E))]

                # I_total_E[0] = I_total_E[0] - np.array(list(map(self._dendritic_function_F, self._excitatory_ampa_current_long_range(s_AMPA))))[0] * amp
                self.rE[:, i, j] = rE + self.rate_dynamics(rE, I_total_E, "E") * self.dt*second

            I_noiseI = self.noise[:, i-1, 2]*amp
            self.noise[:, i, 2] = I_noiseI + self.ornstein_uhlenbeck_process(I_noiseI) * self.dt*second

            s_NDMAo, s_AMPAo = self.sNDMA[:, i-1, :], self.sAMPA[:, i-1, :]

            I_total_I = (
                self._inhibitory_ndma_current_long_range(s_NDMAo)
                + self._inhibitory_ampa_current_long_range(s_AMPAo)
                # + (params.G_I_I * s_GABA)
                + self._inhibitory_ndma_current(s_NDMAo) 
                + I_noiseI + self.params.I_bg_I 
            )

            self.rI[:, i] = rI + self.rate_dynamics(rI, I_total_I, "I") * self.dt*second

        return [self.rE, self.rI, self.sNDMA, self.sAMPA, self.sGABA]

    def run(self):

        init_conditions = np.array((0.1, 0.1, 0.1, # NMDA, AMPA, GABA
                                    0, 0, # rE, rI
                                    0))  # noise,
        

        # Solve the ODE system 
        # result = solve_ivp(self.ode_model, t_span, init_conditions,
        #                    t_eval=t_eval, method='RK45')

        self._init_variables()
        return self.euler_ode_model(init_conditions)

        # print(self.W)
        

    def synaptic_dynamics(self, s, r, tau, gamma, gaba=False):
        if(gaba):
            dsdt = (-s / tau) + gamma * r
        else:
            dsdt = (-s / tau) + (1 - s) * gamma * r
        return dsdt
    
    def rate_dynamics(self, r: float, I_total: int, population: str):
        drdt = (-r + self._threshold_function(population, I_total)) / self.params.tau_r
        return drdt
    
    def ornstein_uhlenbeck_process(self, I):
        dIdt = (-I + np.random.normal(0, 1) * np.sqrt(2 * self.params.sigma_noise**2)) / self.params.tau_AMPA
        return dIdt

    def _threshold_function(self, population: str, I_total: int):

        if(population == "E"):
            val = (self.params.a * I_total - self.params.b)
            return val / (1 - np.exp(-self.params.d * val))

        if(population == "I"):
            return self.params.beta_i * (I_total - self.params.I_th) * (I_total >= self.params.I_th)
        
        logger.error("Crashing... `threshold_function` got bad input for `population` parameter")
        exit()

    def _excitatory_ndma_current(self, S, k_local):
        I = spine_count_gradient_long_range(self.spine_counts, "E") * k_local * self.params.G_n_loc_E_E * S
        return I
    
    def _excitatory_ampa_current(self, S, k_local):
        I = spine_count_gradient_long_range(self.spine_counts, "E") * (1 - k_local) * self.params.G_a_loc_E_E * S
        return I
    
    def _excitatory_gaba_current(self, S):
        I = self.params.G_E_I * S
        return I
    
    def _inhibitory_ndma_current(self, S):
        I = spine_count_gradient_long_range(self.spine_counts, "I") * self.params.G_n_loc_I_E * np.sum(S, axis=1)
        return I
    
    def _excitatory_ndma_current_long_range(self, S):
        sum = 0
        for l in range(self.params.num_of_source_areas):
            sum += self.W[:, l] * (self.sln[:, l]*self.params.k_sup*self.params.p_sup +
                                   (1 - self.sln[:, l])*self.params.k_dp*self.params.p_dp) * S[l]
        
            # print(self.W[:, l])
                                   
        I = self.params.G_E_NMDA * spine_count_gradient_long_range(self.spine_counts, "E") * sum
        return I

    def _excitatory_ampa_current_long_range(self, S):
        sum = 0
        for l in range(self.params.num_of_source_areas):
            sum += self.W[:, l] * (self.sln[:, l]*(1-self.params.k_sup)*self.params.p_sup +
                                   (1 - self.sln[:, l])*(1-self.params.k_dp)*self.params.p_dp) * S[l]
                                   
        I = self.params.G_E_AMPA * spine_count_gradient_long_range(self.spine_counts, "E") * sum
        return I

    def _dendritic_function_F(self, X):
        
        if(X <= 0*pA):
            return 0*pA

        if(X >= 300*pA):
            return 300*pA

        return X

    def _inhibitory_ndma_current_long_range(self, S):
        sum = 0
        for l in range(self.params.num_of_source_areas):
            sum += self.W[:, l] * (self.sln[:, l] * self.params.k_sup * (1-self.params.p_sup) +
                                   (1 - self.sln[:, l]) * self.params.k_dp * (1-self.params.p_dp)) * np.sum(S, axis=1)[l]
                                   
        I = self.params.G_I_NMDA * spine_count_gradient_long_range(self.spine_counts, "I") * sum
        return I

    def _inhibitory_ampa_current_long_range(self, S):
        sum = 0
        for l in range(self.params.num_of_source_areas):
            sum += self.W[:, l] * (self.sln[:, l] * (1-self.params.k_sup) * (1-self.params.p_sup) +
                                   (1 - self.sln[:, l]) * (1-self.params.k_dp) * (1-self.params.p_dp)) * np.sum(S, axis=1)[l]
                                   
        I = self.params.G_I_AMPA * spine_count_gradient_long_range(self.spine_counts, "I") * sum
        return I


    
if( __name__ == "__main__"):

    params = Params(simulation_time=1, dt=0.0001)
    model = LongRangeModel(params)
    result = model.run()

    area = "9/46d"
    area_index = model.area_names.index(area)
    # area_index = model.area_names.index("V1")

    # for i, area in enumerate(model.area_names):
    plt.plot(model.t_span_euler, model.rE[area_index, :, 0], color="red", label="E1")
    plt.plot(model.t_span_euler, model.rE[area_index, :, 1], color="green", label="E2")
    plt.plot(model.t_span_euler, model.rI[area_index], color="blue", label="I")
    plt.title(f"{area}")
    plt.legend()
    plt.show()


    plt.plot(model.t_span_euler, model.sNDMA[23, :, 0], label="NMDA")
    plt.plot(model.t_span_euler, model.sAMPA[23, :, 0], label="AMPA")
    plt.plot(model.t_span_euler, model.sGABA[23, :], label="GABA")
    plt.legend()
    plt.show()
