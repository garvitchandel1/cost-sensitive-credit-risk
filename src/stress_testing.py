
def run_stress_test(proba, stress_factor=1.5):
    import numpy as np
    return np.clip(proba * stress_factor, 0, 1)