# IMPORTS
from email import message
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
import datetime
from getpass import getpass
import socket
from fastkde import fastKDE

from src.models import *
from src.utils import *

#  UTILS
def f_null(t):
    """
    A perturbation function that returns 0. It is used to compute an undisturbed trajectory.

    :param t: time
    :return: 0
    """
    return 0

def index_value(value, array):
    """
    It returns the index of the array element that is closest to the value you pass to the function
    
    :param value: the value you want to find the index of
    :param array: the array you want to search
    :return: The index of the value in the array that is closest to the value.
    """
    idx = (np.abs(array - value)).argmin()
    return idx
#NB : probably the worst interpolation possible but fast... I guess


# STATS
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# shape x : (1,d), shape y : (n,d), return shape : (n,1)
def N(x,y,h):
    """
    The gaussian kernel function as the npFDT paper of cooper.
    
    :param x: the data points. shape x : (1,d).
    :param y: the data point we want to estimate the density at. shape y : (n,d).
    :param h: the bandwidth parameter.
    :return: the value of the kernel function for the given data points x and y. shape : (n,1).
    """
    d = x.shape[1]
    return np.exp(-np.sum((x-y)*(x-y),axis=1)/(2.0*h**2))/(2*np.pi*h**2)**(d/2.0)

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def term_pdf_j_cooper(j, h, x, x_sub=None):
    """
    > The function `term_pdf_j_cooper` the pdf-dependent term at index j for the pdf_mods `cooper` and 'subcooper'.
    
    :param j: the index of the term we're looking at
    :param h: the bandwidth of the kernel
    :param x: the data matrix
    :param x_sub: the subset of the data that we're using to estimate the pdf if pdf_mod='subcooper'
    :return: The ratio between the derivative of the pdf at the point xj and the pdf at xj. shape : (1,d).
    """
    # Used for the pdf_mod 'subcooper'
    if x_sub is not None:
        xj = np.expand_dims(x_sub[j,:],axis=0)
    # Used for the pdf_mod 'cooper'
    else :
        xj = np.expand_dims(x[j,:],axis=0)
    # Compute the derivative of the pdf
    der = np.expand_dims(np.sum((xj-x).T*N(xj,x,h), axis=1), axis=0)
    # Divide by the pdf
    const = np.sum(N(xj,x,h))*(h**2)
    der = der/const
    return der

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def term_pdf_j_fastKDE(j,x,pdf,der_pdf,values):
    """
    >The function `term_pdf_j_fastKDE` the pdf-dependent term at index j for the pdf_mod 'fastKDE'
    
    :param j: the index of the point we're looking at
    :param x: the data points
    :param pdf: the pdf of the data
    :param der_pdf: the derivative of the pdf
    :param values: the values of the pdf at the grid points
    :return: The ratio between the derivative of the pdf at the point xj and the pdf at xj. shape : (1,d).
    """
    nb_dim = x.shape[1]
    xj = np.expand_dims(x[j,:],axis=-1)
    indexes_j=[]
    # Find the indexes of the closest point in the "values" array to compute the pdf at this point.
    # It's in a sense an interpolation of the pdf.
    for dim in range(nb_dim):
        ind = index_value(x[j,dim],values[dim])
        indexes_j.append(ind)
    indexes_j=tuple(indexes_j)
    
    # Put the the derivative of the pdf in the right shape for calculation
    grad_j = np.expand_dims(der_pdf[(slice(None),*indexes_j)], axis=0)

    # Set the term_pdf to 0 if the pdf is to close to 0 at the point xj to avoid errors
    if pdf[indexes_j]<1e-7:
        return grad_j*0.
    else:
        term_j = grad_j/pdf[indexes_j]
    return term_j


def term_pdf_gaussian(x,m):
    """
    > The function `term_pdf_j_fastKDE` the pdf_dependent term at index j for the pdf_mod 'gaussian'
    
    :param x: the data points
    :param m: the number of samples to use for the covariance matrix
    :return: The ratio between the derivative of the pdf at the point xj and the pdf at xj. shape : (1,d).
    """
    # Take the m first samples of the data
    xm = x[:m,:]
    # Compute the covariance matrix and the mean 
    C = np.cov(xm, rowvar=False)
    mu = np.mean(xm, axis=0)
    xbar = xm - mu 
    # Compute the ratio of the derivative of the pdf and the pdf
    if x.shape[1] ==1 :
        invC = 1/C
        term = -invC*xbar
    else :
        invC = np.linalg.inv(C)
        term =  (-invC@xbar[:,:,None])[:,:,0]
    return term


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def lambda_hat(s,m,x,term_pdf):
    """
    > `lambda_hat` computes the value of the matrix $\lambda$ at time lag s.

    
    :param s: the time lag to evaluate lambda at
    :param m: the number of terms in the sum
    :param x: the data
    :param term_pdf: the pdf-dependent term
    :return: lambda_hat is a matrix of size (d x d), with d the number of dimensions of the system.
    """
    xtau=x[s:m+s,:]
    lambd = np.sum(xtau[:,:,None] @ term_pdf[:, None, :], axis=0) / (m*1.)
    return lambd



#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def L_hat(T, dt, x, term_pdf, m, par_mod='bottom'):
    """
    Make the complete estimation of the matrix L in the FDT.
    
    :param T: the total time of the experiment
    :param dt: time step
    :param x: the value of the parameter we're trying to estimate
    :param term_pdf: the pdf-dependent term. np.array of shape (m,d).
    :param m: the number of parameters in the model
    :param par_mod: parallelization mode. Can be 'top' (parallelize over the K iterations) or 'bottom'
    (parallelize over the m values of the trajectory in the estimation of lambda_hat), defaults to top (optional)
    :param test_mod: if True, the pdf will be the perfect gaussian pdf. Otherwise, it will the pdf will 
    be estimated with fastKDE. Defaults to False.
    :return: a (d x d) matrix (with d the number of dimensions of the system) being the integral of the function lambda_hat(s) from 0 to T.
    """

    int_val = np.arange(0,T, dt) # integration values
    # print(int_val) 
    s_val = np.arange(0,len(int_val)).astype(int) # values at which evaluate lambda_hat(s)
    nb_dim = x.shape[-1] 
    # Function to integrate
    if par_mod=='bottom':
        with mp.Pool(mp.cpu_count()) as pool:
            funct_val = np.array(pool.map(partial(lambda_hat,m=m,x=x,term_pdf=term_pdf),s_val))
    else :
        funct_val = np.array(list(map(partial(lambda_hat,m=m,x=x,term_pdf=term_pdf),s_val))) 
    
    
    # Multiply the result by the inverse of lambda_hat(0) to obtain a unbiased estimator of L
    if nb_dim == 1 :
        funct_val = funct_val / lambda_hat(s=0,x=x,term_pdf=term_pdf,m=m)
    else :
        lambda0 = lambda_hat(s=0,x=x,term_pdf=term_pdf,m=m)
        det = np.linalg.det(lambda0)
        if det!=0: # prevent the case of a singular matrix
            funct_val = funct_val @ np.linalg.inv(lambda0)
        else :
            print("SINGULAR MATRIX")
    

    funct_val = np.moveaxis(funct_val,0,-1) # swap axes to have the right shape for np.trapz
    integral = np.trapz(funct_val, int_val) # compute the integral
    return integral #, funct_val, arg_1st_0

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def one_iter(k, model, params, n, m, x0, dt, T, message, password, subsample=True, pre_comp=None, par_mod='bottom', pdf_mod='cooper',  h=0.01, no_mail=False, verbose=False, nb_worker=max(mp.cpu_count()-2,1)):
    """
    It computes one realization of the estimator using the FDT for a given model, given parameters, and given perturbation.
    
    :param k: the iteration number
    :param model: the model to be simulated
    :param params: a dictionary containing the parameters of the model, the dimension of the system, and
    the perturbation function
    :param n: number of points in the trajectory
    :param m: number of points to use in the estimation of the pdf
    :param t0: initial time
    :param tf: the final time of the simulation
    :param x0: initial condition
    :param dt: the time step of the simulation
    :param T: the time of the perturbation
    :param message: a string that will be sent in the email
    :param password: the password of the sender's email address
    :param pre_comp: pre-computed term_pdf (if None, it will be computed)
    :param par_mod: parallelization mode. Can be 'top' or 'bottom'. 'top' means that the parallelization
    is done on the m points of the trajectory. 'bottom' means that the parallelization is done on the k
    iterations, defaults to bottom (optional)
    :param pdf_mod: the method used to estimate the pdf. Can be 'cooper', 'fastKDE', 'subcooper',
    'subcooper_n' or 'gaussian', defaults to cooper (optional)
    :param h: the bandwidth of the kernel used to estimate the pdf in the mods 'cooper' and 'subcooper'
    :param no_mail: if True, no email will be sent, defaults to False (optional)
    :param verbose: if True, will send an email at each iteration, defaults to False (optional)
    :param nb_worker: number of workers to use for parallelization
    :return: The delta_x value
    """
    # Ensure that each process has a different seed
    np.random.seed((os.getpid() * int(time.time())) % 123456789) 

    start = datetime.datetime.now()
    
    # Get model parameters
    model_params = params['model_params']
    nb_dim=params['dim']
    f = params['f']

    # Simulate the trajectory : 
    if subsample:
        dt = dt * 100
        n_actual = n + int(T/dt) 
        if 'linear_model' in str(model):
            # the function fast_one_trajectory_LM is faster than one_trajectory for the linear model
            x = fast_one_trajectory_LM(model_params,delta_t=dt, n=n_actual)
        else :
            x, tspan = one_trajectory(model(model_params, f=f_null), x0=x0, t0=0, tf=n_actual, n=int(n_actual*100))
            # take one point each 100 points to tackle the variance problem (done systematically in the function fast_one_trajectory_LM)
            x = x[::100,:]
    else :
        n_actual = n + int(T/dt)
        x, tspan = one_trajectory(model(model_params, f=f_null), x0=x0, t0=0, tf=n_actual/100, n=n_actual)

    # Compute the pdf is pre_comp is not provided
    if pre_comp is None :

        if pdf_mod=='cooper':
            if par_mod=='top':
                term_pdf = []
                for j in trange(m):
                    term_pdf.append(term_pdf_j_cooper(j=j,h=h,x=x))
            else : 
                with mp.Pool(nb_worker) as pool:
                    term_pdf = pool.map(partial(term_pdf_j_cooper,h=h,x=x), range(m))
            term_pdf = np.array(term_pdf)[:,0,:]
            
        elif pdf_mod=='fastKDE':
            # Estimate the pdf on a subsample of relevant values with fastKDE
            fastKDE_precision = 33 # 129 # previously 257 but too slow
            numPoints=list((np.ones(nb_dim)*fastKDE_precision).astype(int)) 
            pdf, values = fastKDE.pdf(*x.T, numPoints=numPoints)
            if nb_dim==1:
                values = [values]
                der_pdf = np.gradient(pdf, *values)
                der_pdf= der_pdf[None,:]
            else:
                der_pdf = np.gradient(pdf, *values)
                der_pdf=np.stack(der_pdf,axis=0)
            # Compute the pdf-term (ratio of the derivative of the pdf and the pdf)
            if par_mod=='top':
                term_pdf = []
                for j in trange(m):
                    term_pdf.append(term_pdf_j_fastKDE(j=j,x=x,pdf=pdf,der_pdf=der_pdf,values=values))
            else : 
                with mp.Pool(nb_worker) as pool:
                    term_pdf = pool.map(partial(term_pdf_j_fastKDE,x=x,pdf=pdf,der_pdf=der_pdf,values=values), range(m))
            term_pdf = np.array(term_pdf)[:,0,:]

        elif pdf_mod[:9]=='subcooper':
            # Get the number of points in the subsample directly from the name of the pdf_mod
            if pdf_mod[9:]!='':
                n_sub = int(pdf_mod[9:])
            else :
                n_sub = int(1e3)
            # Make sure the size of the subsample is not greater than m
            if n_sub > m :
                n_sub = m
            # 1) Choose the subsample randomly in the trajectory
            x_sub = x[np.random.choice(range(x.shape[0]), n_sub, replace=False),:]
            # 2) Estimate the pdf-dependent terms on the subsample
            if par_mod=='top':
                term_pdf_sub = []
                for j in trange(n_sub):
                    term_pdf_sub.append(term_pdf_j_cooper(j=j,h=h,x=x,x_sub=x_sub))
            else : 
                with mp.Pool(nb_worker) as pool: #To-do : fix to be coherent with previously max(...,1)
                    term_pdf_sub = pool.map(partial(term_pdf_j_cooper,h=h,x=x,x_sub=x_sub), range(n_sub))
            term_pdf_sub = np.array(term_pdf_sub)[:,0,:]
            # 3) Find the closest point in the trajectory to the subsample
            indices = np.linalg.norm(x[:m,:]-x_sub[:,None,:], axis=2).argmin(axis=0)
            term_pdf = term_pdf_sub[indices,:]

        elif pdf_mod=='gaussian':
            term_pdf = term_pdf_gaussian(x,m)

        # Send msg to the user of the time it tooked to compute the pdf
        time_pdf= datetime.datetime.now()
        msg_pdf = f"PDF ESTIMATED at iter {k}... begining of the L_hat estimation after {(time_pdf-start).total_seconds()} seconds"
        print(msg_pdf)
        msg_iter = message + "\n" +  msg_pdf
        if verbose:
            if not no_mail:
                # Send intermediate results to the user
                send_update(subject=f"pdf SIMULATION n={format(n,'.2E')}", message=msg_iter, sender_password=password)

    else :
        term_pdf = pre_comp
        time_pdf= datetime.datetime.now()
        print("PDF PRE-COMPUTED... begining of the L_hat estimation")
    
    # Compute the estimator of L
    L = L_hat(T=T, dt=dt, x=x, term_pdf=term_pdf, m=m, par_mod=par_mod)
    # Multiply the result by the perturbation value (steady case) to obtain delta_x as in the FDT
    if nb_dim==1:
        delta_x=L*f(0)
        # back to 0-dimensional
        delta_x=delta_x[0,0]
    elif nb_dim >1:
        delta_x=L@f(0)
    
    # Display some informations on the result
    if par_mod=='bottom' :
        print("Result iter {}: {}".format(k, delta_x))
        print(f"Computation time after pdf: {(datetime.datetime.now()-time_pdf).total_seconds()} seconds")
    return delta_x


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def simulation(model, params, K, n, m, x0, dt, T, subsample=True, pre_comp=None, par_mod='bottom', pdf_mod='cooper', h=0.01, no_mail=False, verbose=False, password=None, nb_worker=max(mp.cpu_count()-2,1)):
    """
    Compute the full estimation procedure using the FDT.
    
    :param model: the name of the model you want to use
    :param params: a dictionary containing the parameters of the model, the dimension of the model, and
    the perturbation function
    :param K: number of iterations
    :param n: number of samples
    :param m: number of samples to use for the Monte Carlo estimation of the integral
    :param x0: initial condition. Must be choose according to the equilibrium distribution of the model.
    :param t0: initial time
    :param tf: the final time of the simulation
    :param dt: the time step of the simulation
    :param T: the number of time steps*
    :param subsample: if True, the pdf is estimated on a subsample of the trajectory, otherwise the pdf is estimated on the whole trajectory
    :param pre_comp: if you have already computed the pdfs, you can pass them as a parameter
    :param par_mod: parallelization mode. Can be 'top' or 'bottom'. 'top' means that the parallelization
    is done on the iterations of the algorithm, 'bottom' means that the parallelization is done on the
    samples of the algorithm, defaults to bottom (optional)
    :param pdf_mod: how the pdf is computed. Can be 'cooper' or 'gaussian', defaults to cooper (optional)
    :param h: the bandwidth of the kernel used to estimate the pdf in the mods 'cooper' and 'subcooper'
    :param no_mail: if True, no email will be sent, defaults to False (optional). 
    (if you're not Amaury Lancelin, set it to True OR change the function 'send_update' to include your email)
    :param verbose: if True, the function will print the results of each iteration, defaults to False
    (optional)
    :param password: your email password
    :param nb_worker: the number of workers to use for parallelization
    :return: The mean of the results and the results themselves.
    """
    
    start_time = datetime.datetime.now()
    start_time_str = str(start_time)[:-7]

    if not no_mail:
        if password is None:
            email_password = getpass("Type your email password and press enter: ")
        else :
            email_password = password
    if no_mail:
        email_password = ''


    print("---"*25)
    msg_beg_sim = f"Launching simulations with K={K} iterations, n={format(n,'.2E')} samples..."
    print(f"{msg_beg_sim}")
    
    #-----------------------------------------------------------------------------------------------------
    message = "---"*38 + "\n"
    message += "Beginning of the simulation at " + start_time_str + " with the following parameters :"
    message += f"""
    K = {K}
    n = {format(n,'.2E')}
    m = {format(m,'.2E')}
    h = {h}

    model = {model}
    model_params = {params['model_params']}
    dimension = {params['dim']}
    perturbation value = {params['f'](0)}

    T ={T}
    dt ={dt}
    x0 = {x0}
    
    subsample = {subsample}, pre_comp = {pre_comp is not None}, par_mod = {par_mod}, pdf_mod = {pdf_mod}, 
    verbose = {verbose}, no_mail = {no_mail}, nb_worker = {nb_worker}, 


    Computer : {socket.gethostname()}
    """
    message += "---"*38 + "\n"
    msg_params = message
    if not no_mail:
        send_update(subject=f"beg SIMULATION {start_time_str}", message=message, sender_password=email_password)
    #--------------------------------------------------------------------------------------------------------
    K_=np.arange(K)
    if par_mod=='top':
        with mp.Pool(processes=nb_worker) as pool: 
            results = pool.map(partial(one_iter, model=model, params=params, n=n, m=m, h=h,
                                        x0=x0, dt=dt, T=T, message=msg_params, password=email_password, subsample=subsample,
                                        pre_comp=pre_comp, par_mod=par_mod, pdf_mod=pdf_mod, no_mail=no_mail, verbose=verbose), K_)
    else:
        results=[]
        for k in tqdm(K_):
            beg_iter = datetime.datetime.now()
            res = one_iter(model=model, params=params, k=k, n=n, m=m, h=h,
                    x0=x0, dt=dt, T=T, message=msg_params, password=email_password, subsample=subsample, 
                    pre_comp=pre_comp, par_mod=par_mod, pdf_mod=pdf_mod, no_mail=no_mail, verbose=verbose)
            end_iter = datetime.datetime.now()
            if verbose:
                message += f"After {(end_iter-beg_iter).total_seconds()} seconds, result iter {k} : {res} " + "\n"
                if not no_mail:
                    send_update(subject=f"iter {k} SIMULATION {start_time_str}", message=message, sender_password=email_password)
            results.append(res)
    
    # Display some informations on the result
    end_time = datetime.datetime.now()
    if verbose :
        message += "---"*38 + "\n"
    message += f"End of the simulation after {(end_time-start_time).total_seconds()/60.} minutes" + "\n"
    message +="RESULTS : \n"
    nb_dim=int(params['dim'])
    if nb_dim==1:
        message += f"{results}" + "\n" + f"mean = {np.mean(results, axis=0)}" + "\n" + f"std = {np.std(results, axis=0)}"
    else :
        message += f"{results}"
    if not no_mail:
        # Send final results to the user
        send_update(subject=f"end SIMULATION {start_time_str}", message=message, sender_password=email_password)
    
    if nb_dim==1:
        return np.mean(results), results
    else :
        return np.mean(results,axis=0), results




