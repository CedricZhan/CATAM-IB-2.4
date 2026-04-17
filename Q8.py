import numpy as np
import matplotlib.pyplot as plt

T = 10.0
L = 10
rng = np.random.default_rng(42)  

theta_list = [k * np.pi / (2**7) for k in range(1, 2**6 + 1)]

def grad_f(x, theta):
    return 4 * x**3 - 3 * x - np.cos(theta)

def gradient_descent_terminal(x0, theta, h, T=10.0):
    n_steps = int(round(T / h))
    x = x0
    for _ in range(n_steps):
        x = x - h * grad_f(x, theta)
    return x

def sample_Y_l(theta, l, rng, T=10.0):
    x0 = rng.uniform(-1.0, 1.0)

    h_l = 0.1 * (2.0 ** (-l))
    x_fine = gradient_descent_terminal(x0, theta, h_l, T)

    if l == 0:
        return x_fine
    else:
        h_prev = 0.1 * (2.0 ** (-(l - 1)))
        x_coarse = gradient_descent_terminal(x0, theta, h_prev, T)
        return x_fine - x_coarse

def mlmc_estimator(theta, L, Nl_list, rng, T=10.0):
    estimate = 0.0
    level_means = []

    for l in range(L + 1):
        Nl = Nl_list[l]
        samples = np.array([sample_Y_l(theta, l, rng, T) for _ in range(Nl)])
        mean_l = samples.mean()
        estimate += mean_l
        level_means.append(mean_l)

    return estimate, level_means

Nl_const5 = [5] * (L + 1)
Nl_geometric = [2 ** (L - l) for l in range(L + 1)]

print("Case i) N_l ≡ 5")
print("-" * 50)
results_const5 = []
for theta in theta_list:
    mu_hat, level_means = mlmc_estimator(theta, L, Nl_const5, rng, T)
    results_const5.append(mu_hat)
    print(f"theta = {theta:.6f} ({theta/np.pi:.3f}π),  mu_hat = {mu_hat:.10f}")

print("\nCase ii) N_l = 2^(L-l)")
print("-" * 50)
results_geometric = []
for theta in theta_list:
    mu_hat, level_means = mlmc_estimator(theta, L, Nl_geometric, rng, T)
    results_geometric.append(mu_hat)
    print(f"theta = {theta:.6f} ({theta/np.pi:.3f}π),  mu_hat = {mu_hat:.10f}")


theta_vals = np.array(theta_list)

plt.figure()
plt.plot(theta_vals, results_const5, label="Nl ≡ 5")
plt.plot(theta_vals, results_geometric, label="Nl = 2^(L-l)")
plt.xlabel("theta")
plt.ylabel("MLMC estimate")
plt.legend()
plt.title("MLMC estimates vs theta")
plt.show()