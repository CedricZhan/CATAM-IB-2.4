import numpy as np
import matplotlib.pyplot as plt

theta = np.pi / 6
h = 0.01
n_steps = 1000

def f(x, theta):
    return (x**2 - 3/4)**2 - x * np.cos(theta)

def df(x, theta):
    return 4 * x * (x**2 - 3/4) - np.cos(theta)

x0_values = np.array([k / 50 for k in range(-50, 51)])

final_x = []
trajectories = []

for x0 in x0_values:
    x = x0
    path = [x]
    for _ in range(n_steps):
        x = x - h * df(x, theta)
        path.append(x)
    final_x.append(x)
    trajectories.append(path)

final_x = np.array(final_x)

for x0, xT in zip(x0_values, final_x):
    print(f"x0 = {x0: .2f} -> x_1000 = {xT: .6f}")

plt.figure(figsize=(8, 5))
plt.plot(x0_values, final_x, 'o', markersize=4)
plt.xlabel("Initial point $x_0$")
plt.ylabel("Final iterate $x_{1000}$")
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))
for path in trajectories:
    plt.plot(path, alpha=0.4)
plt.xlabel("Iteration")
plt.ylabel("$x_t$")
plt.title("Gradient descent trajectories")
plt.grid(True)
plt.show()