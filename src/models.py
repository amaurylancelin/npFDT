# Imports 
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.linalg as la
import scipy
from tqdm.notebook import tqdm, trange
from src.utils import*

# Linear model dynamic
def linear_model(model_params, f):
    """
    It takes a matrix $ and a function $ and returns a function that takes a state $ and a time
    $ and returns the next state '$ according to the linear model ' = Bx + f(t) + \epsilon$ where
    $\epsilon$ is a random white noise term
    
    :param B: the matrix of coefficients for the linear model
    :param f: a function that takes in a time and returns a vector of the same size as the state
    :param noise: the standard deviation of the noise (gaussian)
    :return: A function that takes in x, t, and dt and returns a vector of length n.
    """
    B=model_params['B']
    noise=model_params['noise']
    
    def dyn(x, t, dt):
        d = B.shape[0]
        return np.array([B@x]) + noise*np.random.randn(d)/np.sqrt(dt) + f(t)
    return dyn


# Initial condition for the linear model 
def init_cond_LM(B,noise, seed=None):
    """
    It solves the Lyapunov equation  + C_0B^T = -Q$ for $, then finds the eigenvalues and
    eigenvectors of $, and then uses the eigenvectors to generate a random vector $ with the
    same covariance as $
    
    :param B: the state transition matrix
    :param noise: the standard deviation of the noise in the system
    :param seed: the seed for the random number generator
    :return: The initial condition for the state vector x.
    """
    Q = noise**2 * np.eye(B.shape[0])
    C0 = scipy.linalg.solve_continuous_lyapunov(B, -Q)
    D, V = np.linalg.eig(C0)
    x0 = V.dot(np.diag(np.sqrt(D))).dot(np.random.randn(B.shape[0]))
    return x0

# Get the equilibrium values for the linear model
def get_equilibrium_lm(B, f, nb_samples=1e5):
    """
    It computes the equilibrium of the linear model by simulating the trajectory
    for a long time
    
    :param B: the matrix of the linear model
    :param f: the perturbation function that defines the dynamics of the system
    :param nb_samples: number of samples to use to estimate the equilibrium values
    :return: The last values of the trajectory.
    """
    x0 = init_cond_LM(B=B,noise=0.1)
    model_params = {'B':B,'noise':0.}
    x, tspan = one_trajectory(linear_model(model_params, f=f), x0=x0, t0=0, tf=nb_samples/100, n=nb_samples)
    return x[-1,:]


# Generate several random linear models
def get_models_lm(d_max=10, seed=None):
    """
    It generates a dictionary of models, where the keys are the dimension of the model, and the values
    are the matrices and perturbations
    
    :param d_max: the maximum dimension of the models to be generated, defaults to 10 (optional)
    :param seed: the seed for the random number generator. If you don't want to use a seed, set it to
    None
    :return: A dictionary of models, where the key is the dimension of the model and the value is a list
    of two elements:
            - The first element is a list of lists, where each list is a row of the matrix A
            - The second element is a list of the perturbation values
    """
    dict_models = {}
    d_max=int(d_max)
    for d in range(1,d_max+1):
        if d==1:
            dict_models['1D'] = [[[-0.5]], [1]]
        else :
            if seed is not None :
                np.random.seed(seed)
            # Generate a random matrix
            eig_val = np.random.uniform(-0.7, -0.3, size=d)
            s = np.diag(eig_val)

            q, _ = la.qr(np.random.rand(d, d))
            stableA = q.T @ s @ q

            # # Generate the perturbation
            pert = np.ones(d) #np.linspace(0.7, 1.3+(1.08**d-1), d)

            # print the matrix
            dict_models[f'{d}D'] = [np.ndarray.tolist(stableA), np.ndarray.tolist(pert)]
    return dict_models
# print(dict_models)