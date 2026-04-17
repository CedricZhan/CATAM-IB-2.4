import numpy as np
import matplotlib.pyplot as plt

T = 10
L = 10
N0 = 4000  

def grad_f(x, theta):
    return 4*x**3 - 3*x - np.cos(theta)

def GD(x0, theta, h):
    n_steps = int(T / h)
    x = x0
    for _ in range(n_steps):
        x = x - h * grad_f(x, theta)
    return x

def mlmc_mu(theta):
    hs = [0.1 * 2**(-l) for l in range(L+1)]
    
    Ns = [max(1, int(N0 * 2**(-1.5 * l))) for l in range(L+1)]
    
    mu_est = 0.0
    
    for l in range(L+1):
        h = hs[l]
        Nl = Ns[l]
        
        if l == 0:
            x0 = np.random.uniform(-1, 1, size=Nl)
            Y = np.array([GD(x, theta, h) for x in x0])
        else:
            hc = hs[l-1]
            x0 = np.random.uniform(-1, 1, size=Nl)
            
            xf = np.array([GD(x, theta, h) for x in x0])
            xc = np.array([GD(x, theta, hc) for x in x0])
            
            Y = xf - xc
        
        mu_est += np.mean(Y)
    
    return mu_est

def m1(theta):
    return np.cos((2*np.pi + theta)/3)   

def m2(theta):
    return np.cos(theta/3)              

thetas = np.array([k * np.pi / (2**7) for k in range(1, 2**6 + 1)])

mu_hats = np.array([mlmc_mu(theta) for theta in thetas])

m1_vals = np.array([m1(theta) for theta in thetas])
m2_vals = np.array([m2(theta) for theta in thetas])

p1 = (m2_vals - mu_hats) / (m2_vals - m1_vals)
p2 = 1 - p1

p1 = np.clip(p1, 0, 1)
p2 = np.clip(p2, 0, 1)


plt.figure(figsize=(8,5))
plt.plot(thetas, p1, label='p1(theta)')
plt.plot(thetas, p2, label='p2(theta)')
plt.xlabel('theta')
plt.ylabel('Probability')
plt.title('MLMC estimates of p1(theta) and p2(theta)')
plt.legend()
plt.grid()
plt.show()