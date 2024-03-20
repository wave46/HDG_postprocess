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

    