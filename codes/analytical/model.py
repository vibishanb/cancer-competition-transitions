#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 21:10:18 2025

@author: ckadelka
"""


import numpy as np
from numba import jit

@jit(nopython=True)
def competition_transition_model(X, t, r_s, r_r, c, d, K, t_s, t_r):
    """
    Simulates a 2-species logistic growth model with asymmetric competition and transition.

    Parameters:
    ----------
    X : array-like
        Initial state vector [s,r], where:
          s : sensitive cell population.
          r : resistant cell population.
    t : float
        Current time step.
    r_s : float
        growth rate of sensitive cell population.
    r_r : float
        growth rate of resistant cell population.
    c : float
        competitve effect of resistant cells on sensitive cells.
    d : float
        competitve effect of sensitive cells on resistant cells.
    K : float
        carrying capacity.
    t_s : float
        transition rate of sensitive cells.
    t_r : float
        transition rate of resistant cells.

    Returns:
    -------
    dX : array-like
        Rate of change [ds, dr].
    """
    
    s, r = X
    dX = np.zeros(2,dtype=np.float64)
    dX[0] = r_s * s * (1 - (s + c * r)/ K) - t_s * s + t_r *r
    dX[1] = r_r * r * (1 - (r + d * s)/ K) + t_s * s - t_r *r
    return dX

@jit(nopython=True)
def competition_transition_model_adaptive_treatment(X, t, r_s, r_r, c, d, K, t_s, t_r, d_s, d_r):
    """
    Simulates a 2-species logistic growth model with asymmetric competition and transition.

    Parameters:
    ----------
    X : array-like
        Initial state vector [s,r], where:
          s : sensitive cell population.
          r : resistant cell population.
    t : float
        Current time step.
    r_s : float
        growth rate of sensitive cell population.
    r_r : float
        growth rate of resistant cell population.
    c : float
        competitve effect of resistant cells on sensitive cells.
    d : float
        competitve effect of sensitive cells on resistant cells.
    K : float
        carrying capacity.
    t_s : float
        transition rate of sensitive cells.
    t_r : float
        transition rate of resistant cells.
    d_s : float
        death rate of sensitive cells.
    d_r : float
        death rate of resistant cells.
        
    Returns:
    -------
    dX : array-like
        Rate of change [ds, dr].
    """
    
    s, r = X
    dX = np.zeros(2,dtype=np.float64)
    dX[0] = r_s * s * (1 - (s + c * r)/ K) - t_s * s + t_r *r - d_s * s
    dX[1] = r_r * r * (1 - (r + d * s)/ K) + t_s * s - t_r *r - d_r * r
    return dX

@jit(nopython=True)
def RK4(func, X0, ts, r_s, r_r, c, d, K, t_s, t_r, dt): 
    """
    Implements the 4th-order Runge-Kutta (RK4) method to solve the model.

    Parameters:
    ----------
    func : callable
        Function to evaluate derivatives (e.g., competition_transition_model).
    X0 : array-like
        Initial state vector [s, r].
    ts : array-like
        Time points for simulation.
    r_s : float
        growth rate of sensitive cell population.
    r_r : float
        growth rate of resistant cell population.
    c : float
        competitve effect of resistant cells on sensitive cells.
    d : float
        competitve effect of sensitive cells on resistant cells.
    K : float
        carrying capacity.
    t_s : float
        transition rate of sensitive cells.
    t_r : float
        transition rate of resistant cells.
    dt : float
        Time step size.

    Returns:
    -------
    X : array-like
        Simulated trajectories of [s, r] at each time step.
    """
    
    nt = len(ts)
    X  = np.zeros((nt, 2),dtype=np.float64)
    X[0,:] = X0
    
    for i in range(nt-1):           
        k1 = func(X[i], ts[i], r_s, r_r, c, d, K, t_s, t_r)
        k2 = func(X[i] + dt*k1/2., ts[i] + dt/2.,r_s, r_r, c, d, K, t_s, t_r)
        k3 = func(X[i] + dt*k2/2., ts[i] + dt/2.,r_s, r_r, c, d, K, t_s, t_r)
        k4 = func(X[i] + dt*k3, ts[i] + dt,r_s, r_r, c, d, K, t_s, t_r)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    return X

@jit(nopython=True)
def RK4_adaptive_therapy(func, X0, ts, 
                         r_s, r_s_treatment, r_r, 
                         c, d, K, 
                         t_s, t_s_treatment, t_r, 
                         d_s, d_s_treatment, d_r, 
                         threshold_treatment_on, threshold_treatment_off,  
                         dt): 
    """
    Implements the 4th-order Runge-Kutta (RK4) method to solve the model.

    Parameters:
    ----------
    func : callable
        Function to evaluate derivatives (e.g., competition_transition_model).
    X0 : array-like
        Initial state vector [s, r].
    ts : array-like
        Time points for simulation.
    r_s : float
        growth rate of sensitive cell population.
    r_s_treatment: float
        growth rate of sensitive cell population under treatment.    
    r_r : float
        growth rate of resistant cell population.
    c : float
        competitve effect of resistant cells on sensitive cells.
    d : float
        competitve effect of sensitive cells on resistant cells.
    K : float
        carrying capacity.
    t_s : float
        transition rate of sensitive cells.
    t_s_treatment: float
        transition rate of sensitive cells under treatment.
    t_r : float
        transition rate of resistant cells.
    d_s : float
        death rate of sensitive cells.
    d_s_treatment: float
        death rate of sensitive cells under treatment.
    d_r : float
        death rate of resistant cells.
    threshold_treatment_on : float
        proportion of total cells (relative to carrying capacity) at which treatment is initiated
    threshold_treatment_off : float
        proportion of total cells (relative to carrying capacity) at which treatment is stopped
    dt : float
        Time step size.

    Returns:
    -------
    X : array-like
        Simulated trajectories of [s, r] at each time step.
    """
    
    nt = len(ts)
    X  = np.zeros((nt, 2),dtype=np.float64)
    X[0,:] = X0
    
    r_s_current = r_s
    t_s_current = t_s
    d_s_current = d_s
    treatment = np.zeros(nt,dtype=np.int16)
    
    for i in range(nt-1):           
        k1 = func(X[i], ts[i], r_s_current, r_r, c, d, K, t_s_current, t_r, d_s_current, d_r)
        k2 = func(X[i] + dt*k1/2., ts[i] + dt/2.,r_s_current, r_r, c, d, K, t_s_current, t_r, d_s_current, d_r)
        k3 = func(X[i] + dt*k2/2., ts[i] + dt/2.,r_s_current, r_r, c, d, K, t_s_current, t_r, d_s_current, d_r)
        k4 = func(X[i] + dt*k3, ts[i] + dt,r_s_current, r_r, c, d, K, t_s_current, t_r, d_s_current, d_r)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
        if sum(X[i+1])>threshold_treatment_on*K and treatment[i]==0: #turn treatment on
            r_s_current = r_s_treatment
            t_s_current = t_s_treatment
            d_s_current = d_s_treatment
            treatment[i+1] = 1
        elif sum(X[i+1])<threshold_treatment_off*K and treatment[i]==1: #turn treatment off
            r_s_current = r_s
            t_s_current = t_s
            d_s_current = d_s
            treatment[i+1] = 0
        else:
            treatment[i+1] = treatment[i]
            
            
    return X, treatment


def simulate( r_s = 1, r_r = 1, c = 0.2, d = 0.2, K = 10000, t_s=0.02, t_r=0.02, s0 = 100, r0 = 100, dt=0.1, t_end=500):
    """
    Computes the dynamics of the SIR model with delayed human behavioral responses and calculates additional outputs.

    Parameters:
    ----------
    r_s : float, default = 1
        growth rate of sensitive cell population.
    r_r : float, default = 1
        growth rate of resistant cell population.
    c : float, default = 0.2
        competitve effect of resistant cells on sensitive cells.
    d : float, default = 0.2
        competitve effect of sensitive cells on resistant cells.
    K : float, default = 10000
        carrying capacity.
    t_s : float, default = 0.02
        transition rate of sensitive cells.
    t_r : float, default = 0.02
        transition rate of resistant cells.
    s0 : float, default = 100
        Initial number of sensitive cells.
    r0 : float, default = 100
        Initial number of resistant cells.        
    dt : float, default=0.1
        Time step size.
    t_end : float, default=500
        End time for the simulation.

    Returns:
    -------
    ts : array-like
        Time points of the simulation.
    solution : array-like
        Simulated trajectories of [s, r] at each time step.
    params : dict
        The specific parameters used in the simulation.

    """
    
    ts = np.linspace(0, t_end, round(t_end / dt) + 1)
    x0 = np.array([s0, r0],dtype=np.float64)
    params = {'r_s' : r_s, 'r_r' : r_r, 'c' : c, 'd' : d, 'K' : K, 't_s' : t_s, 't_r' : t_r, 'dt' : dt}

    #check if any parameters are to be linked to other parameters, check for nested linkage
    n_reduced=1
    while n_reduced>0:
        n_reduced = 0
        for key in params:
            if type(params[key])==str and type(params[params[key]]) in [float,int,bool,np.float64]:
                params[key] = params[params[key]]
                n_reduced += 1

    r_s = params['r_s']
    r_r = params['r_r']
    c = params['c']
    d = params['d']
    K = params['K']
    t_s = params['t_s']
    t_r = params['t_r']
    dt = params['dt']

    solution = RK4(competition_transition_model, x0, ts, r_s, r_r, c, d, K, t_s, t_r, dt)
    return ts, solution, params

def simulate_adaptive_therapy(r_s = 0.05, r_s_treatment = 0.01, r_r = 0.05, 
                              c = 0.2, d = 0.2, K = 10000, 
                              t_s=0.005, t_s_treatment = 0., t_r=0.005, 
                              d_s = 0, d_s_treatment = 0.01, d_r = 0, 
                              threshold_treatment_on = 0.5, threshold_treatment_off = 0.1, 
                              s0 = 100, r0 = 100, 
                              dt=0.1, t_end=2500):
    """
    Computes the dynamics of the SIR model with delayed human behavioral responses and calculates additional outputs.

    Parameters:
    ----------
    r_s : float
        growth rate of sensitive cell population.
    r_s_treatment: float
        growth rate of sensitive cell population under treatment.    
    r_r : float
        growth rate of resistant cell population.
    c : float
        competitve effect of resistant cells on sensitive cells.
    d : float
        competitve effect of sensitive cells on resistant cells.
    K : float
        carrying capacity.
    t_s : float
        transition rate of sensitive cells.
    t_s_treatment: float
        transition rate of sensitive cells under treatment.
    t_r : float
        transition rate of resistant cells.
    d_s : float
        death rate of sensitive cells.
    d_s_treatment: float
        death rate of sensitive cells under treatment.
    d_r : float
        death rate of resistant cells.
    threshold_treatment_on : float
        proportion of total cells (relative to carrying capacity) at which treatment is initiated
    threshold_treatment_off : float
        proportion of total cells (relative to carrying capacity) at which treatment is stopped
    s0 : float, default = 100
        Initial number of sensitive cells.
    r0 : float, default = 100
        Initial number of resistant cells.        
    dt : float, default=0.1
        Time step size.
    t_end : float, default=500
        End time for the simulation.

    Returns:
    -------
    ts : array-like
        Time points of the simulation.
    solution : array-like
        Simulated trajectories of [s, r] at each time step.
    treatment: array-like
        Binary array that describes when therapy occurred.
    params: dict
        The specific parameters used in the simulation.

    """
    
    ts = np.linspace(0, t_end, round(t_end / dt) + 1)
    x0 = np.array([s0, r0],dtype=np.float64)
    params = {'r_s' : r_s, 'r_s_treatment' : r_s_treatment, 'r_r' : r_r, 'c' : c, 'd' : d, 'K' : K, 't_s' : t_s, 't_s_treatment' : t_s_treatment, 't_r' : t_r, 'd_s' : d_s, 'd_s_treatment' : d_s_treatment, 'd_r' : d_r, 'threshold_treatment_on' : threshold_treatment_on, 'threshold_treatment_off' : threshold_treatment_off, 'dt' : dt}

    #check if any parameters are to be linked to other parameters, check for nested linkage
    n_reduced=1
    while n_reduced>0:
        n_reduced = 0
        for key in params:
            if type(params[key])==str and type(params[params[key]]) in [float,int,bool,np.float64]:
                params[key] = params[params[key]]
                n_reduced += 1

    r_s = params['r_s']
    r_s_treatment = params['r_s_treatment']
    r_r = params['r_r']
    c = params['c']
    d = params['d']
    K = params['K']
    t_s = params['t_s']
    t_s_treatment = params['t_s_treatment']
    t_r = params['t_r']
    d_s = params['d_s']
    d_s_treatment = params['d_s_treatment']
    d_r = params['d_r']
    threshold_treatment_on = params['threshold_treatment_on']
    threshold_treatment_off = params['threshold_treatment_off']
    dt = params['dt']
    solution,treatment = RK4_adaptive_therapy(competition_transition_model_adaptive_treatment, 
                                              x0, ts, 
                                              r_s, r_s_treatment, r_r, 
                                              c, d, K, 
                                              t_s, t_s_treatment, t_r, 
                                              d_s, d_s_treatment, d_r, 
                                              threshold_treatment_on, threshold_treatment_off,  
                                              dt)
    #solution,treatment = RK4_adaptive_therapy(competition_transition_model_adaptive_treatment, x0, ts, *params)
    return ts, solution, treatment, params

def plot_time_series(r_s = 1, r_s_treatment = 0.5, r_r = 1, 
                     c = 0.2, d = 0.2, K = 10000, 
                     t_s=0.02, t_s_treatment = 0.02, t_r=0.02, 
                     d_s = 0, d_s_treatment = 0.01, d_r = 0, 
                     threshold_treatment_on = 0.5, threshold_treatment_off = 0.1, 
                     s0 = 1000, r0 = 1000, 
                     dt=1, t_end=500,folder_name='figs'):
    import matplotlib.pyplot as plt
    from utils import params_to_text, get_suffix
    import os
    
    if not os.path.exists(folder_name) and len(folder_name)>0:
        os.makedirs(folder_name)
    
    ts, results, treatment, params = simulate_adaptive_therapy(r_s = r_s,r_s_treatment=r_s_treatment,r_r=r_r,
                                                       c=c,d=d,K=K,t_s=t_s,t_s_treatment=t_s_treatment,t_r=t_r,
                                                       d_s=d_s,d_s_treatment=d_s_treatment,d_r=d_r,
                                                       threshold_treatment_on=threshold_treatment_on,threshold_treatment_off=threshold_treatment_off,
                                                       s0=s0,r0=r0,dt=dt,t_end=t_end)
    s = results[:, 0]
    r = results[:, 1]
    proportion_sensitive_cells = s/(s+r)
    
    fig, ax = plt.subplots(figsize=(5,3))
    ax.semilogy(ts, r, linestyle='-', label='resistant')
    ax.plot(ts, s, linestyle='-', label='sensitive')
    ax.legend(frameon=False,loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)
    ax2 = ax.twinx()
    ax.spines[['top']].set_visible(False)
    ax2.spines[['top']].set_visible(False)
    ax2.plot(ts, proportion_sensitive_cells, color='k', linestyle='--')
    dummy = np.where(treatment==1)[0]
    [y1,y2] = ax2.get_ylim()
    ax2.plot(ts[dummy],[y1 + 0.01*(y2-y1)] * len(dummy),'ko')
    ax.set_xlabel('time')
    ax.set_ylabel('number of cells')
    ax2.set_ylabel('proportion sensitive cells')
    text_left,text_right = params_to_text(params)
    ax2.text(np.percentile(ts,15),y2 + 0.2*(y2-y1),'\n'.join(text_left),va='bottom',ha='center')
    ax2.text(np.percentile(ts,85),y2 + 0.2*(y2-y1),'\n'.join(text_right),va='bottom',ha='center')
    ax2.set_ylim([y1,y2])
    suffix = get_suffix(text_left,text_right) + '_tend'+str(t_end)        
    plt.savefig(os.path.join(folder_name,'dynamics'+suffix+'.pdf'), bbox_inches='tight')


if __name__ == "__main__":
    #basic plot: no transitions, no competition, therapy only leads to non-zero death rate of sensitive cells, no other effect
    plot_time_series(r_s = 0.01,r_s_treatment=0.01,r_r = 0.01, t_s = 0.0, t_r = 0.0, t_s_treatment=0.0, d_s_treatment = 0.1, threshold_treatment_off = 0.25, t_end = 400)

    #the presence of transitions enables the emergence of oscillations
    plot_time_series(r_s = 0.01,r_s_treatment=0.01,r_r = 0.01, t_s = 0.01, t_r = 0.01, t_s_treatment=0.01, d_s_treatment = 0.1, threshold_treatment_off = 0.25, t_end = 600)
    
    #As the transition rate increases, the frequency increases.
    plot_time_series(r_s = 0.01,r_s_treatment=0.01,r_r = 0.01, t_s = 0.05, t_r = 0.05, t_s_treatment=0.05, d_s_treatment = 0.1, threshold_treatment_off = 0.25, t_end = 600)

    #At the same time, the frequency decreases if the therapy is stopped at a lower total cell count.
    plot_time_series(r_s = 0.01,r_s_treatment=0.01,r_r = 0.01, t_s = 0.05, t_r = 0.05, t_s_treatment=0.05, d_s_treatment = 0.1, threshold_treatment_off = 0.1, t_end = 600)
    
    #A competitive advantage of sensitive cells does not appear to change much while c < 1.
    plot_time_series(c=0.2, d=2, r_s = 0.05,r_s_treatment=0.05,r_r = 0.05, t_s = 0.0, t_r = 0.0, t_s_treatment=0.0, d_s_treatment = 0.1, threshold_treatment_off = 0.1, t_end = 400)
    plot_time_series(c=0.2, d=2, r_s = 0.05,r_s_treatment=0.05,r_r = 0.05, t_s = 0.001, t_r = 0.001, t_s_treatment=0.001, d_s_treatment = 0.1, threshold_treatment_off = 0.1, t_end = 600)
    
    #For both c and d > 1, 
    plot_time_series(c=2, d=4, r_s = 0.05,r_s_treatment=0.05,r_r = 0.05, t_s = 0.0, t_r = 0.0, t_s_treatment=0.0, d_s_treatment = 0.1, threshold_treatment_off = 0.1, t_end = 400)
    plot_time_series(c=2, d=4, r_s = 0.05,r_s_treatment=0.05,r_r = 0.05, t_s = 0.001, t_r = 0.001, t_s_treatment=0.001, d_s_treatment = 0.1, threshold_treatment_off = 0.1, t_end = 600)
    
    #If therapy pushes sensitive cells to develop resistance (likely the case), osciallations slow down and eventually vanish
    #This is the case because resistant cell numbers no longer decrease at the onset of therapy and thus therapy never stops (stopping criterion won't be reached)
    # plot_time_series(r_s = 0.01,r_s_treatment=0.01,r_r = 0.01, t_s = 0.01, t_r = 0.01, t_s_treatment=0.01, d_s_treatment = 0.1, threshold_treatment_off = 0.25, t_end = 1000)
    # plot_time_series(r_s = 0.01,r_s_treatment=0.01,r_r = 0.01, t_s = 0.01, t_r = 0.01, t_s_treatment=0.02, d_s_treatment = 0.1, threshold_treatment_off = 0.25, t_end = 1000)
    # plot_time_series(r_s = 0.01,r_s_treatment=0.01,r_r = 0.01, t_s = 0.01, t_r = 0.01, t_s_treatment=0.04, d_s_treatment = 0.1, threshold_treatment_off = 0.25, t_end = 1000)
    
    #Constant dose therapy
    plot_time_series(r_s = 0.01,r_s_treatment=0.01,r_r = 0.01, t_s = 0.01, t_r = 0.01, t_s_treatment=0.01, d_s_treatment = 0.01, threshold_treatment_on=0, threshold_treatment_off = np.nan, t_end = 600)
    plot_time_series(r_s = 0.01,r_s_treatment=0.01,r_r = 0.01, t_s = 0.01, t_r = 0.01, t_s_treatment=0.01, d_s_treatment = 0.025, threshold_treatment_on=0, threshold_treatment_off = np.nan, t_end = 600)
    plot_time_series(r_s = 0.01,r_s_treatment=0.01,r_r = 0.01, t_s = 0.01, t_r = 0.01, t_s_treatment=0.01, d_s_treatment = 0.1, threshold_treatment_on=0, threshold_treatment_off = np.nan, t_end = 600)

    plot_time_series(r_s = 0.01,r_s_treatment=0.0001,r_r = 0.01, t_s = 0.01, t_r = 0.01, t_s_treatment=0.01, d_s_treatment = 0, threshold_treatment_on=0, threshold_treatment_off = np.nan, t_end = 600)
    plot_time_series(r_s = 0.01,r_s_treatment=0.001,r_r = 0.01, t_s = 0.01, t_r = 0.01, t_s_treatment=0.01, d_s_treatment = 0, threshold_treatment_on=0, threshold_treatment_off = np.nan, t_end = 600)
    plot_time_series(r_s = 0.01,r_s_treatment=0.005,r_r = 0.01, t_s = 0.01, t_r = 0.01, t_s_treatment=0.01, d_s_treatment = 0, threshold_treatment_on=0, threshold_treatment_off = np.nan, t_end = 600)

    #Adaptive therapy
    #A low death rate does not yield oscillations because resistant cell numbers do not decrease sufficiently at therapy onset
    #-> ratio between death rate and transition rate determines the qualitative behavior
    plot_time_series(r_s = 0.01,r_s_treatment=0.01,r_r = 0.01, t_s = 0.01, t_r = 0.01, t_s_treatment=0.01, d_s_treatment = 0.01, threshold_treatment_off = 0.25, t_end = 600)
    plot_time_series(r_s = 0.01,r_s_treatment=0.01,r_r = 0.01, t_s = 0.01, t_r = 0.01, t_s_treatment=0.01, d_s_treatment = 0.025, threshold_treatment_off = 0.25, t_end = 600)
    plot_time_series(r_s = 0.01,r_s_treatment=0.01,r_r = 0.01, t_s = 0.01, t_r = 0.01, t_s_treatment=0.01, d_s_treatment = 0.1, threshold_treatment_off = 0.25, t_end = 600)

    # #Oscillations for lower doses of cytotoxic treatment with stronger competition
    # plot_time_series(c=2, d=4, r_s = 0.1,r_s_treatment=0.1,r_r = 0.1, t_s = 0.01, t_r = 0.01, t_s_treatment=0.01, d_s_treatment = 0.01, threshold_treatment_off = 0.25, t_end = 600)
    # plot_time_series(c=2, d=2, r_s = 0.1,r_s_treatment=0.1,r_r = 0.1, t_s = 0.01, t_r = 0.01, t_s_treatment=0.01, d_s_treatment = 0.025, threshold_treatment_off = 0.25, t_end = 600)
    # plot_time_series(c=4, d=2, r_s = 0.1,r_s_treatment=0.1,r_r = 0.1, t_s = 0.01, t_r = 0.01, t_s_treatment=0.01, d_s_treatment = 0.1, threshold_treatment_off = 0.25, t_end = 600)

    #Cytostatic treatment by itself does not give oscillations
    plot_time_series(r_s = 0.01,r_s_treatment=0.0001,r_r = 0.01, t_s = 0.01, t_r = 0.01, t_s_treatment=0.01, d_s_treatment = 0, threshold_treatment_off = 0.25, t_end = 600)
    plot_time_series(r_s = 0.01,r_s_treatment=0.001,r_r = 0.01, t_s = 0.01, t_r = 0.01, t_s_treatment=0.01, d_s_treatment = 0, threshold_treatment_off = 0.25, t_end = 600)
    plot_time_series(r_s = 0.01,r_s_treatment=0.005,r_r = 0.01, t_s = 0.01, t_r = 0.01, t_s_treatment=0.01, d_s_treatment = 0, threshold_treatment_off = 0.25, t_end = 600)

    
    
    
    

