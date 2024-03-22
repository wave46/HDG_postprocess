import numpy as np
from .atomic import calculate_cx_rate,calculate_cx_rate_cons,calculate_iz_rate,calculate_iz_rate_cons

def calculate_plasma_resistivity_cons(solutions,Mref,mD,n0,L0,t0,ohmic_coef):
    """
    calculates plasma resistivity based on conservatives values
    """

    if len(solutions.shape)>2:
        eta = ohmic_coef*n0*mD*L0**2/t0**3/(2/3/Mref*solutions[:,:,3]/solutions[:,:,0])**(3/2)
    else:
        eta = ohmic_coef*n0*mD*L0**2/t0**3/(2/3/Mref*solutions[:,3]/solutions[:,0])**(3/2)

    return eta

def calculate_ohmic_source_cons(solutions,jtor,Mref,mD,n0,L0,t0,ohmic_coef):
    """
    calculates phmic heating source resistivity based on conservatives values
    """


    eta = calculate_plasma_resistivity_cons(solutions,Mref,mD,n0,L0,t0,ohmic_coef)
    return eta*jtor**2

def calculate_variable_cons(solutions,variable,adimensionalization,cons_idx,physics=None,gradients=None,
                            dnn_params=None,atomic_parameters=None):
    """
    for a given variable calculates it based on conservative variables
    """
    defined_variables = ['n','nn','te','ti','M','dnn','mfp','u',
                             'p_dyn','q_i_par','q_e_par','gamma',
                             'q_i_par_conv','q_i_par_cond',
                             'q_e_par_conv','q_e_par_cond']

    defined_variables = ['n','nn','u','Ei']
    
    if variable not in defined_variables:
        raise KeyError(f'{variable} is not in the list of posible variables: {defined_variables}')
    
    if variable == 'n': 
        return calculate_n_cons(solutions,adimensionalization['density_scale'],cons_idx)
    elif variable == 'nn':
        return calculate_nn_cons(solutions,adimensionalization['density_scale'],cons_idx)
    elif variable == 'u':
        return calculate_u_cons(solutions,adimensionalization['speed_scale'],cons_idx)
    elif variable == 'Ei':
        return calculate_Ei_cons(solutions,adimensionalization['mass_scale']*adimensionalization['speed_scale']**2,cons_idx)
    elif variable == 'Ee':
        return calculate_Ee_cons(solutions,adimensionalization['mass_scale']*adimensionalization['speed_scale']**2,cons_idx)
    else:
        raise KeyError(f'{variable} is not in the list of posible variables: {defined_variables}')


def calculate_n_cons(solutions,n0,cons_idx):
    """
    calculates plasma density value based on conservatives values
    """
    if len(solutions.shape)>2:
        res = solutions[:,:,cons_idx[b'rho']]*n0
    else:
        res = solutions[:,cons_idx[b'rho']]*n0
    return res

def calculate_nn_cons(solutions,n0,cons_idx):
    """
    calculates plasma density value based on conservatives values
    """
    if len(solutions.shape)>2:
        res = solutions[:,:,cons_idx[b'rhon']]*n0
    else:
        res = solutions[:,cons_idx[b'rhon']]*n0
    return res

def calculate_u_cons(solutions,u0,cons_idx):
    """
    calculates parallel velocity value based on conservatives values
    """
    dimensions = None
    if len(solutions.shape)>2:
        dimensions = solutions.shape
        sol = solutions.reshape(solutions.shape[0]*solutions.shape[1],solutions.shape[2])

    res = u0*sol[:,cons_idx[b'gamma']]/sol[:,cons_idx[b'rho']]
    if dimensions is not None:
        res = res.reshape(dimensions[0],dimensions[1])
    return res

def calculate_Ei_cons(solutions,E0,cons_idx):
    """
    calculates ion energy value based on conservatives values
    """
    dimensions = None
    if len(solutions.shape)>2:
        dimensions = solutions.shape
        sol = solutions.reshape(solutions.shape[0]*solutions.shape[1],solutions.shape[2])

    res = E0*sol[:,cons_idx[b'nEi']]/sol[:,cons_idx[b'rho']]
    if dimensions is not None:
        res = res.reshape(dimensions[0],dimensions[1])
    return res

def calculate_Ee_cons(solutions,E0,cons_idx):
    """
    calculates electron energy value based on conservatives values
    """
    dimensions = None
    if len(solutions.shape)>2:
        dimensions = solutions.shape
        sol = solutions.reshape(solutions.shape[0]*solutions.shape[1],solutions.shape[2])

    res = E0*sol[:,cons_idx[b'nEe']]/sol[:,cons_idx[b'rho']]
    if dimensions is not None:
        res = res.reshape(dimensions[0],dimensions[1])
    return res