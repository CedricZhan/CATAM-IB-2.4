import numpy as np
import pandas as pd

theta = np.pi / 4
T = 10.0
cos_theta = np.cos(theta)

def grad_f(x, theta):
    return 4 * x**3 - 3 * x - np.cos(theta)

def run_gradient_descent_samples(h, N, theta=np.pi/4, T=10.0, seed=0):

    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=N)
    n_steps = int(round(T / h))

    for _ in range(n_steps):
        x = x - h * grad_f(x, theta)

    return x

rows = []

for k in range(11):
    h = 0.1 * 2**(-k)
    N = 2**(20 - k)

    x_terminal = run_gradient_descent_samples(h=h, N=N, theta=theta, T=T, seed=1234 + k)

    mu_hat = np.mean(x_terminal)
    sample_var = np.var(x_terminal, ddof=1)

    rows.append({
        "k": k,
        "h": h,
        "N_k": N,
        "mu_hat": mu_hat,
        "sample_var_of_X": sample_var
    })

df = pd.DataFrame(rows)
print(df)