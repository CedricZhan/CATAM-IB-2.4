# README

# Project: Sensitivity of Optimisation Algorithms to Initialisation

This project contains Python code for the computational questions in the CATAM project **2.4 Sensitivity of Optimisation Algorithms to Initialisation**.

The project studies gradient descent on a one-dimensional double-well objective function, and uses Monte Carlo and Multi-Level Monte Carlo methods to investigate how the final outcome depends on the initial point and the parameter theta.

# Programming Language

Python 3

# Required Libraries

numpy  
matplotlib  
scipy.optimize  

# Objective Function

The main function is:

f_theta(x) = (x^2 - 3/4)^2 - x cos(theta),    x in [-1, 1]

Gradient descent is applied using:

x_{t+1} = x_t - h f'_theta(x_t)

# File Structure

Q2.py  
- Runs gradient descent for theta = pi/6 from many initial points in [-1, 1].
- Shows how different initialisations converge to different local minima.

Q4.py  
- Uses Monte Carlo simulation to estimate the expected final position of gradient descent for theta = pi/4.
- Compares different step sizes h and studies how the estimate and variance change.

Q8.py  
- Implements the Multi-Level Monte Carlo estimator for different theta values.
- Compares two choices of level sample sizes: N_l = 5 and N_l = 2^(L-l).

Q13.py  
- Uses an MLMC scheme to estimate the probabilities of converging to each local minimum.
- Plots how these probabilities change with theta.

README.md  
- Describes the project structure and explains what each code file does.

# How to Run

Run each file independently:

python Q2.py  
python Q4.py  
python Q8.py  
python Q13.py  

# Notes

All parameters are defined inside the scripts.

No external input files are required.

Plots are generated directly when running the scripts.

Monte Carlo results may vary slightly between runs because of random sampling.