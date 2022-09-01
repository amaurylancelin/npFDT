#Imports 
import numpy as np
import matplotlib.pyplot as plt
import scipy
from tqdm.notebook import trange
from tqdm import tqdm
import multiprocessing as mp
import threading as th
from functools import partial
import os
import time

# email
import email
import smtplib
import ssl
from getpass import getpass
    
    
from src.models import *



# DYNAMICS 
# Simulate the trajectory for a given dynamics
def one_trajectory(dynamics, x0, t0=0, tf=100, n=1000):
    """
    Simulate the trajectory of a given dynamical system.
    :param dynamics: a function that computes the derivative of the state
    :param x0: initial state
    :param t0: initial time
    :param tf: final time
    :param n: number of samples in the trajectory
    :return: state trajectory
    """
    tspan = np.linspace(t0, tf, int(n))
    dt = tspan[1] - tspan[0]
    # initialize the state trajectory
    x = np.zeros((len(tspan), len(x0)))
    # set the initial state
    x[0, :] = x0
    # simulate the trajectory
    for i in range(1, len(tspan)):
        x[i, :] = x[i-1, :] + dt * dynamics(x[i-1, :], tspan[i], dt)
    return x, tspan

def fast_one_trajectory_LM(model_params, delta_t=1, n=1000):
    """ Generates a time series of length run_length of the linear stochastic system (B,noise) in a faster way that one_trajectory
        dx/dt = B.x + r where <r.r'>=Q=noise^2*identy(d)
        using the Euler-Maruyama method
        x_{n+1} = x_n + B*x_n*dt + V*sqrt(D)*randn(len(B))*sqrt(dt)
        where [V,D]=eig(Q) and dt=delta_t/100. The initial condition is
        drawn from the appropriate Gaussian distribution to correctly "spin up".
    
    :param x  - The state vector of the system
    :param r  - A white noise vector with covariance Q
    :param B  - The decay matrix (real part of eigenvalues must all be negative)
    :param Q  - The noise covariance matrix (Must be positive definite)
    :param dt - The time step used
    :param delta_t   - Time between recording data points
    :param n - Record runLength data points """

    save_N_step = 100           # Only save 1/100 time steps
    dt = delta_t / save_N_step  # dt is the timestep used to integrate

    B=model_params['B']
    noise=model_params['noise']
    Q = noise**2 * np.identity(B.shape[0])

    matrix_size = len(B)    # The number of variables in the system

    # Check B
    D = np.real(np.linalg.eig(B)[0])
    if (np.max(D) > 0):
        print('ERROR in fast_one_trajectory_LM:')
        print('Eigenvalues of B have positive real parts.')
        print('Stochastic system explodes, giving up...')
        x = 0
        return x

    # Check Q
    D = np.linalg.eig(Q)[0]
    if (np.min(D) < 0):
        print('ERROR in fast_one_trajectory_LM:')
        print('Eigenvalues of Q are negative.')
        print('Random numbers are imaginary, giving up...')
        x = 0
        return x

    # Optimisation variables
    Qd, Qv = np.linalg.eig(Q)
    Q_opt = Qv.dot(np.diag(np.sqrt(Qd))) * np.sqrt(dt)
    B_opt = np.eye(matrix_size) + B * dt

    # Allocate RAM
    x = np.zeros((n, matrix_size))

    # Initial condition
    C0 = scipy.linalg.solve_continuous_lyapunov(B, -Q)
    D, V = np.linalg.eig(C0)
    x[0,:] = V.dot(np.diag(np.sqrt(D))).dot(np.random.randn(matrix_size))

    # Run the model
    for n in range(n-1):
        temp = x[n,:]
        r = Q_opt.dot(np.random.randn(save_N_step,matrix_size).T).T
        for m in range(save_N_step):
            # temp=temp+B*temp*dt+Qv*sqrt(Qd)*randn(matrix_size)*sqrt(dt);
            temp = B_opt.dot(temp.T) + r[m,:]
        
        x[n+1,:] = temp
    
    return x


# OTHER
def send_update(subject, message, sender_password=None, sender="amaury-lancelin@hotmail.fr", recipient="amaury-lancelin@hotmail.fr"):
    """
    > The function `send_update` takes as input a subject, a message and a password (optional) and sends
    an email to the recipient with the given subject and message
    
    :param subject: The subject of the email
    :param message: The message to be sent
    :param sender_password: The password of the sender's email address
    """
    
    port = 587  
    smtp_server = "smtp-mail.outlook.com"
    if sender_password is None :
        sender_password =   getpass("Type your email password and press enter: ")
    
    msg = email.message_from_string(f'{message}')
    msg['From'] = "amaury-lancelin@hotmail.fr"
    msg['To'] = "amaury-lancelin@hotmail.fr"
    msg['Subject'] = subject

    SSL_context = ssl.create_default_context()
    with smtplib.SMTP(smtp_server, port) as server:
        server.starttls(context=SSL_context)
        server.login(sender, sender_password)
        server.sendmail(sender, recipient, msg.as_string())



def rename_images(path_folder='outputs/archived_plots'):
    """
    It renames all the files in a folder by replacing the colon character with a semicolon
    
    :param path_folder: the folder where the images are stored, defaults to outputs/archived_plots
    (optional)
    """
    for count, filename in enumerate(os.listdir(path_folder)):
        if ':' in filename:
            new_name = filename.replace(':', ';')
            print(f'{filename} -> {new_name}')
            src =f"{path_folder}/{filename}"  # foldername/filename, if .py file is outside folder
            dst =f"{path_folder}/{new_name}"
            os.rename(src, dst)
