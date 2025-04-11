import numpy as np
import matplotlib.pyplot as plt

from LocalModel import LocalModel
from ParamsLocal import Params

params = Params()
model = LocalModel()

res = 200

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

nullclineE = np.vectorize(lambda sE, sI : model.synaptic_dynamics(sE, rateFuncE(sE, sI), params.tau_NMDA, params.gamma_NMDA))(sNDMA_grid, sGABA_grid) 

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
plt.xlim([-0.05, 1.05])
plt.title('Phase Potrait')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.show()

