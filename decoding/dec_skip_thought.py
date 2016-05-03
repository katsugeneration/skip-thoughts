import tools
import numpy as np

dec = tools.load_model()
vec = np.load("../training/vecs.npz")['arr_0']

for cond in vec:
    text = tools.run_sampler(dec, cond, stochastic=True)
    print text