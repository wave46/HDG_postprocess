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

def calculate_grad_n_cons(gradients,n0,L0,cons_idx):
    """
    calculates gradient of plasma density value based on conservatives values
    """
    dimensions = None
    if len(gradients.shape)>3:
        dimensions = gradients.shape
        grad = gradients.reshape(gradients.shape[0]*gradients.shape[1],gradients.shape[2],gradients.shape[3])
    else:
        grad = gradients.copy()

    res = grad[:,cons_idx[b'rho'],:]*n0/L0
    if dimensions is not None:
        res = res.reshape(dimensions[0],dimensions[1],dimensions[2],dimensions[3])
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

def calculate_grad_nn_cons(gradients,n0,L0,cons_idx):
    """
    calculates gradient of neutral density value based on conservatives values
    """
    dimensions = None
    if len(gradients.shape)>3:
        dimensions = gradients.shape
        grad = gradients.reshape(gradients.shape[0]*gradients.shape[1],gradients.shape[2],gradients.shape[3])
    else:
        grad = gradients.copy()

    res = grad[:,cons_idx[b'rhon'],:]*n0/L0
    if dimensions is not None:
        res = res.reshape(dimensions[0],dimensions[1],dimensions[2],dimensions[3])
    return res

def calculate_u_cons(solutions,u0,cons_idx):
    """
    calculates parallel velocity value based on conservatives values
    """
    dimensions = None
    if len(solutions.shape)>2:
        dimensions = solutions.shape
        sol = solutions.reshape(solutions.shape[0]*solutions.shape[1],solutions.shape[2])
    else:
        sol = solutions.copy()

    res = u0*sol[:,cons_idx[b'Gamma']]/sol[:,cons_idx[b'rho']]
    if dimensions is not None:
        res = res.reshape(dimensions[0],dimensions[1])
    return res

def calculate_grad_u_cons(solutions,gradients,u0,L0,cons_idx):
    """
    calculates gradient of plasma velocity value based on conservatives values
    """
    dimensions = None
    if len(gradients.shape)>3:
        dimensions = gradients.shape
        grad = gradients.reshape(gradients.shape[0]*gradients.shape[1],gradients.shape[2],gradients.shape[3])
        sol = solutions.reshape(solutions.shape[0]*solutions.shape[1],solutions.shape[2])
    else:
        grad = gradients.copy()
        sol = solutions.copy()

    res = (grad[:,cons_idx[b'rho'],:]*(-1*sol[:,cons_idx[b'Gamma']][:,None]/sol[:,cons_idx[b'rho']][:,None]**2)+
          grad[:,cons_idx[b'Gamma'],:]/sol[:,cons_idx[b'rho']][:,None])*u0/L0
    if dimensions is not None:
        res = res.reshape(dimensions[0],dimensions[1],dimensions[2],dimensions[3])
    return res

def calculate_Ei_cons(solutions,E0,cons_idx):
    """
    calculates ion energy value based on conservatives values
    """
    dimensions = None
    if len(solutions.shape)>2:
        dimensions = solutions.shape
        sol = solutions.reshape(solutions.shape[0]*solutions.shape[1],solutions.shape[2])
    else:
        sol = solutions.copy()
    res = E0*sol[:,cons_idx[b'nEi']]/sol[:,cons_idx[b'rho']]
    if dimensions is not None:
        res = res.reshape(dimensions[0],dimensions[1],dimensions[2])
    return res

def calculate_grad_Ei_cons(solutions,gradients,E0,L0,cons_idx):
    """
    calculates gradient of ion energy value based on conservatives values
    """
    dimensions = None
    if len(gradients.shape)>3:
        dimensions = gradients.shape
        grad = gradients.reshape(gradients.shape[0]*gradients.shape[1],gradients.shape[2],gradients.shape[3])
        sol = solutions.reshape(solutions.shape[0]*solutions.shape[1],solutions.shape[2])
    else:
        grad = gradients.copy()
        sol = solutions.copy()

    res = (grad[:,cons_idx[b'rho'],:]*(-1*sol[:,cons_idx[b'nEi']][:,None]/sol[:,cons_idx[b'rho']][:,None]**2)+
          grad[:,cons_idx[b'nEi'],:]/sol[:,cons_idx[b'rho']][:,None])*E0/L0
    if dimensions is not None:
        res = res.reshape(dimensions[0],dimensions[1],dimensions[2],dimensions[3])
    return res

def calculate_Ee_cons(solutions,E0,cons_idx):
    """
    calculates electron energy value based on conservatives values
    """
    dimensions = None
    if len(solutions.shape)>2:
        dimensions = solutions.shape
        sol = solutions.reshape(solutions.shape[0]*solutions.shape[1],solutions.shape[2])
    else:
        sol = solutions.copy()

    res = E0*sol[:,cons_idx[b'nEe']]/sol[:,cons_idx[b'rho']]
    if dimensions is not None:
        res = res.reshape(dimensions[0],dimensions[1],dimensions[2])
    return res

def calculate_grad_Ee_cons(solutions,gradients,E0,L0,cons_idx):
    """
    calculates gradient of electron energy value based on conservatives values
    """
    dimensions = None
    if len(gradients.shape)>3:
        dimensions = gradients.shape
        grad = gradients.reshape(gradients.shape[0]*gradients.shape[1],gradients.shape[2],gradients.shape[3])
        sol = solutions.reshape(solutions.shape[0]*solutions.shape[1],solutions.shape[2])
    else:
        grad = gradients.copy()
        sol = solutions.copy()

    res = (grad[:,cons_idx[b'rho'],:]*(-1*sol[:,cons_idx[b'nEe']][:,None]/sol[:,cons_idx[b'rho']][:,None]**2)+
          grad[:,cons_idx[b'nEe'],:]/sol[:,cons_idx[b'rho']][:,None])*E0/L0
    if dimensions is not None:
        res = res.reshape(dimensions[0],dimensions[1],dimensions[2],dimensions[3])
    return res

def calculate_pi_cons(solutions,p0,cons_idx):
    """
    calculates ion pressure value based on conservatives values
    """
    dimensions = None
    if len(solutions.shape)>2:
        dimensions = solutions.shape
        sol = solutions.reshape(solutions.shape[0]*solutions.shape[1],solutions.shape[2])
    else:
        sol = solutions.copy()

    res = p0*(sol[:,cons_idx[b'nEi']]-0.5*sol[:,cons_idx[b'Gamma']]**2/sol[:,cons_idx[b'rho']])
    if dimensions is not None:
        res = res.reshape(dimensions[0],dimensions[1],dimensions[2])
    return res

def calculate_grad_pi_cons(solutions,gradients,p0,L0,cons_idx):
    """
    calculates gradient of ion pressure value based on conservatives values
    """
    dimensions = None
    if len(gradients.shape)>3:
        dimensions = gradients.shape
        grad = gradients.reshape(gradients.shape[0]*gradients.shape[1],gradients.shape[2],gradients.shape[3])
        sol = solutions.reshape(solutions.shape[0]*solutions.shape[1],solutions.shape[2])
    else:
        grad = gradients.copy()
        sol = solutions.copy()

    res = grad[:,cons_idx[b'nEi'],:]
    res -= grad[:,cons_idx[b'Gamma'],:]*sol[:,cons_idx[b'Gamma']][:,None]/sol[:,cons_idx[b'rho']][:,None]
    res += 0.5*grad[:,cons_idx[b'rho'],:]*sol[:,cons_idx[b'Gamma']][:,None]**2/sol[:,cons_idx[b'rho']][:,None]**2
    res *= p0/L0

    if dimensions is not None:
        res = res.reshape(dimensions[0],dimensions[1],dimensions[2],dimensions[3])
    return res

def calculate_pe_cons(solutions,p0,cons_idx):
    """
    calculates electron pressure value based on conservatives values
    """
    dimensions = None
    if len(solutions.shape)>2:
        dimensions = solutions.shape
        sol = solutions.reshape(solutions.shape[0]*solutions.shape[1],solutions.shape[2])
    else:
        sol = solutions.copy()

    res = p0*(sol[:,cons_idx[b'nEe']])
    if dimensions is not None:
        res = res.reshape(dimensions[0],dimensions[1],dimensions[2])
    return res

def calculate_grad_pe_cons(gradients,p0,L0,cons_idx):
    """
    calculates gradient of electron pressure value based on conservatives values
    """
    dimensions = None
    if len(gradients.shape)>3:
        dimensions = gradients.shape
        grad = gradients.reshape(gradients.shape[0]*gradients.shape[1],gradients.shape[2],gradients.shape[3])
    else:
        grad = gradients.copy()

    res = grad[:,cons_idx[b'nEe'],:]
    res *= p0/L0

    if dimensions is not None:
        res = res.reshape(dimensions[0],dimensions[1],dimensions[2],dimensions[3])
    return res

def calculate_pdyn_cons(solutions,p0,E0,cons_idx):
    """
    calculates dynamic pressure value based on conservatives values
    kb(Ti+Te)+mD*u**2, first is dimensionalized by p0, second by m0*n0*u0**2
    """
    dimensions = None
    if len(solutions.shape)>2:
        dimensions = solutions.shape
        sol = solutions.reshape(solutions.shape[0]*solutions.shape[1],solutions.shape[2])
    else:
        sol = solutions.copy()
    pe = calculate_pe_cons(sol,1,cons_idx)
    pi = calculate_pi_cons(sol,1,cons_idx)
    u = calculate_u_cons(sol,1,cons_idx)
    n = calculate_n_cons(sol,1,cons_idx)
    res = p0*(pe+pi)+E0*n*u**2
    if dimensions is not None:
        res = res.reshape(dimensions[0],dimensions[1])
    return res

def calculate_Ti_cons(solutions,T0,Mref,cons_idx):
    """
    calculates ion temperature value based on conservatives values

    """
    dimensions = None
    if len(solutions.shape)>2:
        dimensions = solutions.shape
        sol = solutions.reshape(solutions.shape[0]*solutions.shape[1],solutions.shape[2])
    else:
        sol = solutions.copy()

    res = 2/3/Mref*T0*(sol[:,cons_idx[b'nEi']]/sol[:,cons_idx[b'rho']]-
              0.5*sol[:,cons_idx[b'Gamma']]**2/sol[:,cons_idx[b'rho']]**2)
    if dimensions is not None:
        res = res.reshape(dimensions[0],dimensions[1])
    return res

def calculate_grad_Ti_cons(solutions,gradients,T0,Mref,L0,cons_idx):
    """
    calculates gradient of ion temerature value based on conservatives values
    """
    dimensions = None
    if len(gradients.shape)>3:
        dimensions = gradients.shape
        grad = gradients.reshape(gradients.shape[0]*gradients.shape[1],gradients.shape[2],gradients.shape[3])
        sol = solutions.reshape(solutions.shape[0]*solutions.shape[1],solutions.shape[2])
    else:
        grad = gradients.copy()
        sol = solutions.copy()
    #grad(Ti) = T0/L0*2/3/Mref*(grad(U1)*(U2**2/U1**3-U3/U1**2)+grad(U2)*(-1*U2/U1**2)+grad(U3)/U1)

    res = grad[:,cons_idx[b'nEi'],:]/sol[:,cons_idx[b'rho']][:,None]
    res += grad[:,cons_idx[b'rho'],:]*(sol[:,cons_idx[b'Gamma']]**2/sol[:,cons_idx[b'rho']]**3-
                                       sol[:,cons_idx[b'nEi']]/sol[:,cons_idx[b'rho']]**2)[:,None]
    res -= grad[:,cons_idx[b'Gamma'],:]*(sol[:,cons_idx[b'Gamma']]/sol[:,cons_idx[b'rho']]**2)[:,None]
    res *= 2/3/Mref*T0/L0

    if dimensions is not None:
        res = res.reshape(dimensions[0],dimensions[1],dimensions[3])
    return res

def calculate_grad_Ti_par_cons(solutions,gradients,Br,Bz,Bt,T0,Mref,L0,cons_idx):
    """
    calculates parallel gradient of ion temerature value based on conservatives values
    """
    dimensions = None
    if len(gradients.shape)>3:
        dimensions = gradients.shape
        grad = gradients.reshape(gradients.shape[0]*gradients.shape[1],gradients.shape[2],gradients.shape[3])
        sol = solutions.reshape(solutions.shape[0]*solutions.shape[1],solutions.shape[2])
        Br_res = Br.flatten()
        Bz_res = Bz.flatten()
        Bt_res = Bt.flatten()
    else:
        grad = gradients.copy()
        sol = solutions.copy()
        Br_res = Br.copy()
        Bz_res = Bz.copy()
        Bt_res = Bt.copy()
    #grad(Ti)_par = (grad(Ti),b)

    grad_ti = calculate_grad_Ti_cons(sol,grad,T0,Mref,L0,cons_idx)
    br = Br_res/np.sqrt(Br_res**2+Bz_res**2+Bt_res**2)
    bz = Bz_res/np.sqrt(Br_res**2+Bz_res**2+Bt_res**2)

    res = grad_ti[:,0]*br + grad_ti[:,1]*bz 

    if dimensions is not None:
        res = res.reshape(dimensions[0],dimensions[1])
    return res

def calculate_Te_cons(solutions,T0,Mref,cons_idx):
    """
    calculates electron temperature value based on conservatives values
    """
    dimensions = None
    if len(solutions.shape)>2:
        dimensions = solutions.shape
        sol = solutions.reshape(solutions.shape[0]*solutions.shape[1],solutions.shape[2])
    else:
        sol = solutions.copy()

    res = 2/3/Mref*T0*(sol[:,cons_idx[b'nEe']]/sol[:,cons_idx[b'rho']])
    if dimensions is not None:
        res = res.reshape(dimensions[0],dimensions[1])
    return res

def calculate_grad_Te_cons(solutions,gradients,T0,Mref,L0,cons_idx):
    """
    calculates gradient of electron temerature value based on conservatives values
    """
    dimensions = None
    if len(gradients.shape)>3:
        dimensions = gradients.shape
        grad = gradients.reshape(gradients.shape[0]*gradients.shape[1],gradients.shape[2],gradients.shape[3])
        sol = solutions.reshape(solutions.shape[0]*solutions.shape[1],solutions.shape[2])
    else:
        grad = gradients.copy()
        sol = solutions.copy()

    res = (grad[:,cons_idx[b'rho'],:]*(-1*sol[:,cons_idx[b'nEe']][:,None]/sol[:,cons_idx[b'rho']][:,None]**2)+
          grad[:,cons_idx[b'nEe'],:]/sol[:,cons_idx[b'rho']][:,None])*T0/L0*2/3/Mref


    if dimensions is not None:
        res = res.reshape(dimensions[0],dimensions[1],dimensions[3])
    return res

def calculate_grad_Te_par_cons(solutions,gradients,Br,Bz,Bt,T0,Mref,L0,cons_idx):
    """
    calculates parallel gradient of electron temerature value based on conservatives values
    """
    dimensions = None
    if len(gradients.shape)>3:
        dimensions = gradients.shape
        grad = gradients.reshape(gradients.shape[0]*gradients.shape[1],gradients.shape[2],gradients.shape[3])
        sol = solutions.reshape(solutions.shape[0]*solutions.shape[1],solutions.shape[2])
        Br_res = Br.flatten()
        Bz_res = Bz.flatten()
        Bt_res = Bt.flatten()
    else:
        grad = gradients.copy()
        sol = solutions.copy()
        Br_res = Br.copy()
        Bz_res = Bz.copy()
        Bt_res = Bt.copy()
    #grad(Ti)_par = (grad(Ti),b)

    grad_te = calculate_grad_Te_cons(sol,grad,T0,Mref,L0,cons_idx)
    br = Br_res/np.sqrt(Br_res**2+Bz_res**2+Bt_res**2)
    bz = Bz_res/np.sqrt(Br_res**2+Bz_res**2+Bt_res**2)

    res = grad_te[:,0]*br + grad_te[:,1]*bz 

    if dimensions is not None:
        res = res.reshape(dimensions[0],dimensions[1])
    return res

def calculate_cs_cons(solutions,u0,cons_idx):
    """
    calculates sound speed of the plasma value based on conservatives values
    """
    dimensions = None
    if len(solutions.shape)>2:
        dimensions = solutions.shape
        sol = solutions.reshape(solutions.shape[0]*solutions.shape[1],solutions.shape[2])
    else:
        sol = solutions.copy()
    #cs = sqrt(2/3*(U3+U4-1/2*U2**2/U1)/U1)

    res = u0*np.sqrt(2/3*(sol[:,cons_idx[b'nEi']]+sol[:,cons_idx[b'nEe']]-
                     0.5*sol[:,cons_idx[b'Gamma']]**2/sol[:,cons_idx[b'rho']])/sol[:,cons_idx[b'rho']])
    if dimensions is not None:
        res = res.reshape(dimensions[0],dimensions[1],dimensions[2])
    return res

def calculate_grad_cs_cons(solutions,gradients,u0,L0,cons_idx):
    """
    calculates gradient of sound speed value based on conservatives values
    cs = u0*(2/3*(U3+U4-1/2*U2**2/U1)/U1)**0.5
    grad(cs) = u0/L0/2/(cs/u0)*(2/3)*(grad(U1)*(-U3/U1**2-U4/U1**2+U2**2/U1**3)+
                                      grad(U2)*(-U2/U1**2)+grad(U3)/U1+grad(U4)/U1)
    """
    dimensions = None
    if len(gradients.shape)>3:
        dimensions = gradients.shape
        grad = gradients.reshape(gradients.shape[0]*gradients.shape[1],gradients.shape[2],gradients.shape[3])
        sol = solutions.reshape(solutions.shape[0]*solutions.shape[1],solutions.shape[2])
    else:
        grad = gradients.copy()
        sol = solutions.copy()

    cs = calculate_cs_cons(sol,u0,cons_idx)/u0

    res = grad[:,cons_idx[b'rho'],:]*((sol[:,cons_idx[b'Gamma']]**2/sol[:,cons_idx[b'rho']]-
                                      (sol[:,cons_idx[b'nEe']]+sol[:,cons_idx[b'nEi']]))/sol[:,cons_idx[b'rho']])[:,None]
    res -= grad[:,cons_idx[b'Gamma'],:]*(sol[:,cons_idx[b'Gamma']]/sol[:,cons_idx[b'rho']]**2)[:,None]
    res += (grad[:,cons_idx[b'nEi'],:]+grad[:,cons_idx[b'nEe'],:])/sol[:,cons_idx[b'rho']][:,None]
    res *=(1/3/cs[:,None])
    res *= u0*L0


    if dimensions is not None:
        res = res.reshape(dimensions[0],dimensions[1],dimensions[2],dimensions[3])
    return res

def calculate_M_cons(solutions,cons_idx):
    """
    calculates Mach number value based on conservatives values
    """
    dimensions = None
    if len(solutions.shape)>2:
        dimensions = solutions.shape
        sol = solutions.reshape(solutions.shape[0]*solutions.shape[1],solutions.shape[2])
    else:
        sol = solutions.copy()
    #cs = sqrt(2/3*(U3+U4-1/2*U2**2/U1)/U1)
    u = calculate_u_cons(sol,1,cons_idx)
    cs = calculate_cs_cons(sol,1,cons_idx)
    res = u/cs
    if dimensions is not None:
        res = res.reshape(dimensions[0],dimensions[1],dimensions[2])
    return res

def calculate_grad_M_cons(solutions,gradients,L0,cons_idx):
    """
    calculates gradient of Mach number value based on conservatives values

    """
    dimensions = None
    if len(gradients.shape)>3:
        dimensions = gradients.shape
        grad = gradients.reshape(gradients.shape[0]*gradients.shape[1],gradients.shape[2],gradients.shape[3])
        sol = solutions.reshape(solutions.shape[0]*solutions.shape[1],solutions.shape[2])
    else:
        grad = gradients.copy()
        sol = solutions.copy()

    cs = calculate_cs_cons(sol,1,cons_idx)[:,None]
    grad_cs = calculate_grad_cs_cons(sol,grad,1,1,cons_idx)
    u = calculate_u_cons(sol,1,cons_idx)[:,None]
    grad_u = calculate_grad_u_cons(sol,grad,1,1,cons_idx)

    res = grad_u/cs-grad_cs*u/cs**2
    res /= L0


    if dimensions is not None:
        res = res.reshape(dimensions[0],dimensions[1],dimensions[2],dimensions[3])
    return res

def calculate_k_cons(solutions,k0,cons_idx):
    """
    calculates turbulent energy value based on conservatives values
    """
    dimensions = None
    if len(solutions.shape)>2:
        dimensions = solutions.shape
        sol = solutions.reshape(solutions.shape[0]*solutions.shape[1],solutions.shape[2])
    else:
        sol = solutions.copy()

    res = sol[:,cons_idx[b'k']]*k0
    if dimensions is not None:
        res = res.reshape(dimensions[0],dimensions[1],dimensions[2])
    return res

def calculate_grad_k_cons(gradients,k0,L0,cons_idx):
    """
    calculates gradient of plasma density value based on conservatives values
    """
    dimensions = None
    if len(gradients.shape)>3:
        dimensions = gradients.shape
        grad = gradients.reshape(gradients.shape[0]*gradients.shape[1],gradients.shape[2],gradients.shape[3])
    else:
        grad = gradients.copy()

    res = grad[:,cons_idx[b'k'],:]*k0/L0
    if dimensions is not None:
        res = res.reshape(dimensions[0],dimensions[1],dimensions[2],dimensions[3])
    return res

def calculate_dk_cons(solutions,dk_params,q_cyl,R,D0,cons_idx):
    """
    calculates turbulent energy diffusion value based on conservatives values
    R adimensional
    """
    dimensions = None
    if len(solutions.shape)>2:
        dimensions = solutions.shape
        sol = solutions.reshape(solutions.shape[0]*solutions.shape[1],solutions.shape[2])
        R_res = R.flatten()
        q_res = q_cyl.flatten()
    else:
        sol = solutions.copy()
        R_res = R.copy()
        q_res = q_cyl.copy()

    cs = calculate_cs_cons(sol,1,cons_idx)
    k = calculate_k_cons(sol,1,cons_idx)

    res = 2*np.pi*R_res*q_res*k/cs
    res[np.isnan(cs)] = dk_params['dk_min']
    res[cs<1e-20] = dk_params['dk_min']
    res[res<dk_params['dk_min_adim']] = dk_params['dk_min_adim']
    res[res>dk_params['dk_max_adim']] = dk_params['dk_max_adim']
    if dimensions is not None:
        res = res.reshape(dimensions[0],dimensions[1])
    return res*D0

def calculate_q_cyl(R,Br,Bz,Bt,a):
    """
    calculates q_cyl for points with given major radii R, magnetic filed and minor radii a
    """

    q_cyl = np.abs(Bt)*a/np.sqrt(Br**2+Bz**2)/R
    q_cyl[q_cyl<1] = 1
    q_cyl[q_cyl>1e4] = 1e4
    
    return q_cyl

def calculate_a(vertices,r_axis,z_axis):
    """
    calculates q_cyl for given verices
    """   
    dimensions = None
    if len(vertices.shape)>2:
        dimensions = vertices.shape
        vert = vertices.reshape(vertices.shape[0]*vertices.shape[1],vertices.shape[2])
    else:
        vert = vertices.copy()
    res =  np.sqrt((vert[:,0]-r_axis)**2+(vert[:,1]-z_axis)**2)
    if dimensions is not None:
        res = res.reshape(dimensions[0],dimensions[1])
    return res

def calculate_parallel_flux_cons(solutions,gamma0,cons_idx):
    """
    calculates parallel velocity value based on conservatives values
    """
    dimensions = None
    if len(solutions.shape)>2:
        dimensions = solutions.shape
        sol = solutions.reshape(solutions.shape[0]*solutions.shape[1],solutions.shape[2])
    else:
        sol = solutions.copy()

    res = gamma0*sol[:,cons_idx[b'Gamma']]
    if dimensions is not None:
        res = res.reshape(dimensions[0],dimensions[1])
    return res

def calculate_parallel_ion_heat_flux_par_conv_cons(solutions,n0,T0,Mref,kb,mD,u0,cons_idx):
    """
    calculates parallel convective ion heat flux value based on conservatives values
    q_ipar = (5/2*kb*n*Ti+1/2*mD*n*u**2)u
    """
    dimensions = None
    if len(solutions.shape)>2:
        dimensions = solutions.shape
        sol = solutions.reshape(solutions.shape[0]*solutions.shape[1],solutions.shape[2])
    else:
        sol = solutions.copy()

    u = calculate_u_cons(sol,u0,cons_idx)
    ti = calculate_Ti_cons(sol,T0,Mref,cons_idx)
    n = calculate_n_cons(sol,n0,cons_idx)
    res = n*u*(5/2*kb*ti+0.5*mD*u**2)
    if dimensions is not None:
        res = res.reshape(dimensions[0],dimensions[1])
    return res

def calculate_parallel_ion_heat_flux_par_cond_cons(solutions,gradients,Br,Bz,Bt,q0,T0,Mref,L0,Tmax,cons_idx):
    """
    calculates parallel conductive ion heat flux value based on conservatives values
    q_ipar = - kappa_par_i*Ti**(5/2)*dTi/dl = q0*Ti**(5/2)*dTi/dl
    """
    dimensions = None
    if len(solutions.shape)>2:
        dimensions = gradients.shape
        sol = solutions.reshape(solutions.shape[0]*solutions.shape[1],solutions.shape[2])
        grad = gradients.reshape(gradients.shape[0]*gradients.shape[1],gradients.shape[2],gradients.shape[3])
        Br_res = Br.reshape(Br.shape[0]*Br.shape[1])
        Bz_res = Bz.reshape(Bz.shape[0]*Bz.shape[1])
        Bt_res = Bt.reshape(Bt.shape[0]*Bt.shape[1])
    else:
        sol = solutions.copy()
        grad = gradients.copy()
        Br_res =Br.copy()
        Bz_res =Bz.copy()
        Bt_res =Bt.copy()

    ti = calculate_Ti_cons(sol,T0,Mref,cons_idx)
    ti = np.minimum(Tmax,ti)
    grad_ti_par = calculate_grad_Ti_par_cons(sol,grad,Br_res,Bz_res,Bt_res,T0,Mref,L0,cons_idx)
    res = -q0*ti**(5/2)*grad_ti_par
    if dimensions is not None:
        res = res.reshape(dimensions[0],dimensions[1])
    return res


def calculate_parallel_ion_heat_flux_par_cons(solutions,gradients,Br,Bz,Bt,n0,q0,T0,Mref,kb,mD,u0,L0,Tmax,cons_idx):
    """
    calculates parallel conductive ion heat flux value based on conservatives values
    q_ipar = q_parcond+q_iparconv
    """
    dimensions = None
    if len(solutions.shape)>2:
        dimensions = solutions.shape
        sol = solutions.reshape(solutions.shape[0]*solutions.shape[1],solutions.shape[2])
        grad = gradients.reshape(gradients.shape[0]*gradients.shape[1],gradients.shape[2],gradients.shape[3])
        Br_res = Br.reshape(Br.shape[0]*Br.shape[1])
        Bz_res = Bz.reshape(Bz.shape[0]*Bz.shape[1])
        Bt_res = Bt.reshape(Bt.shape[0]*Bt.shape[1])
    else:
        sol = solutions.copy()
        grad = gradients.copy()
        Br_res =Br.copy()
        Bz_res =Bz.copy()
        Bt_res =Bt.copy()

    q_iconv = calculate_parallel_ion_heat_flux_par_conv_cons(sol,n0,T0,Mref,kb,mD,u0,cons_idx)
    q_icond = calculate_parallel_ion_heat_flux_par_cond_cons(sol,grad,Br_res,Bz_res,Bt_res,q0,T0,Mref,L0,Tmax,cons_idx)

    res = q_iconv+q_icond
    if dimensions is not None:
        res = res.reshape(dimensions[0],dimensions[1])
    return res


def calculate_parallel_electron_heat_flux_par_conv_cons(solutions,n0,T0,Mref,kb,u0,cons_idx):
    """
    calculates parallel convective electron heat flux value based on conservatives values
    q_epar = (5/2*kb*n*Te)u
    """
    dimensions = None
    if len(solutions.shape)>2:
        dimensions = solutions.shape
        sol = solutions.reshape(solutions.shape[0]*solutions.shape[1],solutions.shape[2])
    else:
        sol = solutions.copy()

    u = calculate_u_cons(sol,u0,cons_idx)
    te= calculate_Te_cons(sol,T0,Mref,cons_idx)
    n = calculate_n_cons(sol,n0,cons_idx)
    res = n*u*(5/2*kb*te)
    if dimensions is not None:
        res = res.reshape(dimensions[0],dimensions[1])
    return res

def calculate_parallel_electron_heat_flux_par_cond_cons(solutions,gradients,Br,Bz,Bt,q0,T0,Mref,L0,Tmax,cons_idx):
    """
    calculates parallel conductive electron heat flux value based on conservatives values
    q_epar = - kappa_par_e*Te**(5/2)*dTe/dl= q0*Te**(5/2)*dTe/dl
    """
    dimensions = None
    if len(solutions.shape)>2:
        dimensions = gradients.shape
        sol = solutions.reshape(solutions.shape[0]*solutions.shape[1],solutions.shape[2])
        grad = gradients.reshape(gradients.shape[0]*gradients.shape[1],gradients.shape[2],gradients.shape[3])
        Br_res = Br.reshape(Br.shape[0]*Br.shape[1])
        Bz_res = Bz.reshape(Bz.shape[0]*Bz.shape[1])
        Bt_res = Bt.reshape(Bt.shape[0]*Bt.shape[1])
    else:
        sol = solutions.copy()
        grad = gradients.copy()
        Br_res =Br.copy()
        Bz_res =Bz.copy()
        Bt_res =Bt.copy()

    te = calculate_Te_cons(sol,T0,Mref,cons_idx)
    te = np.minimum(Tmax,te)
    grad_te_par = calculate_grad_Te_par_cons(sol,grad,Br_res,Bz_res,Bt_res,T0,Mref,L0,cons_idx)
    res = -q0*te**(5/2)*grad_te_par
    if dimensions is not None:
        res = res.reshape(dimensions[0],dimensions[1])
    return res


def calculate_parallel_electron_heat_flux_par_cons(solutions,gradients,Br,Bz,Bt,n0,q0,T0,Mref,kb,u0,L0,Tmax,cons_idx):
    """
    calculates parallel conductive electron heat flux value based on conservatives values
    q_epar = q_parcond+q_iparconv
    """
    dimensions = None
    if len(solutions.shape)>2:
        dimensions = gradients.shape
        sol = solutions.reshape(solutions.shape[0]*solutions.shape[1],solutions.shape[2])
        grad = gradients.reshape(gradients.shape[0]*gradients.shape[1],gradients.shape[2],gradients.shape[3])
        Br_res = Br.reshape(Br.shape[0]*Br.shape[1])
        Bz_res = Bz.reshape(Bz.shape[0]*Bz.shape[1])
        Bt_res = Bt.reshape(Bt.shape[0]*Bt.shape[1])
    else:
        sol = solutions.copy()
        grad = gradients.copy()
        Br_res =Br.copy()
        Bz_res =Bz.copy()
        Bt_res =Bt.copy()

    q_econv = calculate_parallel_electron_heat_flux_par_conv_cons(sol,n0,T0,Mref,kb,u0,cons_idx)
    q_econd = calculate_parallel_electron_heat_flux_par_cond_cons(sol,grad,Br_res,Bz_res,Bt_res,q0,T0,Mref,L0,Tmax,cons_idx)

    res = q_econv+q_econd
    if dimensions is not None:
        res = res.reshape(dimensions[0],dimensions[1])
    return res

def calculate_particle_perp_flux_wall_cons(solutions,gradients,diffusion,Br,Bz,Bt,n,n0,L0,cons_idx):
    """
    calculates perpendicular particle flux on the wall with normal n value based on conservatives values
    diffusion dimensional (assuming Dperp only)
    """
    dimensions = None
    if len(solutions.shape)>2:
        dimensions = gradients.shape
        sol = solutions.reshape(solutions.shape[0]*solutions.shape[1],solutions.shape[2])
        grad = gradients.reshape(gradients.shape[0]*gradients.shape[1],gradients.shape[2],gradients.shape[3])
        Br_res = Br.reshape(Br.shape[0]*Br.shape[1])
        Bz_res = Bz.reshape(Bz.shape[0]*Bz.shape[1])
        Bt_res = Bt.reshape(Bt.shape[0]*Bt.shape[1])
        n_res = n.reshape(n.shape[0]*n.shape[1],n.shape[2])
        diffusion_res = diffusion.reshape(diffusion.shape[0]*diffusion.shape[1])
    else:
        sol = solutions.copy()
        grad = gradients.copy()
        Br_res =Br.copy()
        Bz_res =Bz.copy()
        Bt_res =Bt.copy()
        n_res = n.copy()
        diffusion_res = diffusion.copy()

    grad_n = calculate_grad_n_cons(grad,n0,L0,cons_idx)

    br = Br_res/np.sqrt(Br_res**2+Bz_res**2+Bt_res**2)
    bz = Br_res/np.sqrt(Br_res**2+Bz_res**2+Bt_res**2)
    bn = br*n_res[:,0]+bz*n_res[:,1]

    res = -1*diffusion_res*(grad_n[:,0]*n_res[:,0]+grad_n[:,1]*n_res[:,1]-grad_n[:,0]*bn*br-grad_n[:,1]*bn*bz)

    if dimensions is not None:
        res = res.reshape(dimensions[0],dimensions[1])
    return res

def calculate_perp_ion_heat_wall_cons(solutions,gradients,diffusion,Br,Bz,Bt,n,E0,L0,cons_idx):
    """
    calculates perpendicular ion heat flux on the wall with normal n value based on conservatives values
    diffusion dimensional (assuming Dperp only)
    """
    dimensions = None
    if len(solutions.shape)>2:
        dimensions = gradients.shape
        sol = solutions.reshape(solutions.shape[0]*solutions.shape[1],solutions.shape[2])
        grad = gradients.reshape(gradients.shape[0]*gradients.shape[1],gradients.shape[2],gradients.shape[3])
        Br_res = Br.reshape(Br.shape[0]*Br.shape[1])
        Bz_res = Bz.reshape(Bz.shape[0]*Bz.shape[1])
        Bt_res = Bt.reshape(Bt.shape[0]*Bt.shape[1])
        n_res = n.reshape(n.shape[0]*n.shape[1],n.shape[2])
        diffusion_res = diffusion.reshape(diffusion.shape[0]*diffusion.shape[1])
    else:
        sol = solutions.copy()
        grad = gradients.copy()
        Br_res =Br.copy()
        Bz_res =Bz.copy()
        Bt_res =Bt.copy()
        n_res = n.copy()
        diffusion_res = diffusion.copy()

    

    br = Br_res/np.sqrt(Br_res**2+Bz_res**2+Bt_res**2)
    bz = Br_res/np.sqrt(Br_res**2+Bz_res**2+Bt_res**2)
    bn = br*n_res[:,0]+bz*n_res[:,1]
    grad_nEi = grad[:,cons_idx[b'nEi']]

    res = -1*diffusion_res*(grad_nEi[:,0]*n_res[:,0]+grad_nEi[:,1]*n_res[:,1]-grad_nEi[:,0]*bn*br-grad_nEi[:,1]*bn*bz)*E0/L0

    if dimensions is not None:
        res = res.reshape(dimensions[0],dimensions[1])
    return res

def calculate_perp_electron_heat_wall_cons(solutions,gradients,diffusion,Br,Bz,Bt,n,E0,L0,cons_idx):
    """
    calculates perpendicular ion heat flux on the wall with normal n value based on conservatives values
    diffusion dimensional (assuming Dperp only)
    """
    dimensions = None
    if len(solutions.shape)>2:
        dimensions = gradients.shape
        sol = solutions.reshape(solutions.shape[0]*solutions.shape[1],solutions.shape[2])
        grad = gradients.reshape(gradients.shape[0]*gradients.shape[1],gradients.shape[2],gradients.shape[3])
        Br_res = Br.reshape(Br.shape[0]*Br.shape[1])
        Bz_res = Bz.reshape(Bz.shape[0]*Bz.shape[1])
        Bt_res = Bt.reshape(Bt.shape[0]*Bt.shape[1])
        n_res = n.reshape(n.shape[0]*n.shape[1],n.shape[2])
        diffusion_res = diffusion.reshape(diffusion.shape[0]*diffusion.shape[1])
    else:
        sol = solutions.copy()
        grad = gradients.copy()
        Br_res =Br.copy()
        Bz_res =Bz.copy()
        Bt_res =Bt.copy()
        n_res = n.copy()
        diffusion_res = diffusion.copy()

    

    br = Br_res/np.sqrt(Br_res**2+Bz_res**2+Bt_res**2)
    bz = Br_res/np.sqrt(Br_res**2+Bz_res**2+Bt_res**2)
    bn = br*n_res[:,0]+bz*n_res[:,1]
    grad_nEe = grad[:,cons_idx[b'nEe']]

    res = -1*diffusion_res*(grad_nEe[:,0]*n_res[:,0]+grad_nEe[:,1]*n_res[:,1]-grad_nEe[:,0]*bn*br-grad_nEe[:,1]*bn*bz)*E0/L0

    if dimensions is not None:
        res = res.reshape(dimensions[0],dimensions[1])
    return res

