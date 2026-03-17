import numpy as np


def _flatten_solution_nodes(solutions):
    if solutions.ndim > 2:
        shape = solutions.shape[:2]
        return solutions.reshape(shape[0] * shape[1], solutions.shape[2]), shape
    return solutions, None


def _flatten_gradient_nodes(gradients):
    if gradients.ndim > 3:
        shape = gradients.shape[:2]
        return gradients.reshape(shape[0] * shape[1], gradients.shape[2], gradients.shape[3]), shape
    return gradients, None


def _flatten_solution_gradient_nodes(solutions, gradients):
    grad, shape = _flatten_gradient_nodes(gradients)
    if shape is None:
        return solutions, grad, None
    sol = solutions.reshape(shape[0] * shape[1], solutions.shape[2])
    return sol, grad, shape


def _flatten_scalar_field(field):
    if field.ndim > 1:
        shape = field.shape[:2]
        return field.reshape(shape[0] * shape[1]), shape
    return field, None


def _flatten_parallel_inputs(Br, Bz, Bt):
    Br_flat, shape = _flatten_scalar_field(Br)
    Bz_flat, _ = _flatten_scalar_field(Bz)
    Bt_flat, _ = _flatten_scalar_field(Bt)
    return Br_flat, Bz_flat, Bt_flat, shape


def _reshape_scalar(values, shape):
    if shape is None:
        return values
    return values.reshape(shape)


def _reshape_vector(values, shape):
    if shape is None:
        return values
    return values.reshape(shape[0], shape[1], values.shape[1])


def _sol(solutions, key, cons_idx):
    return solutions[:, cons_idx[key]]


def _grad(gradients, key, cons_idx):
    return gradients[:, cons_idx[key], :]


def _ratio_gradient(grad_numerator, grad_denominator, numerator, denominator):
    return grad_numerator / denominator[:, None] - grad_denominator * (
        numerator / denominator**2
    )[:, None]


def _field_aligned_components(Br, Bz, Bt):
    norm = np.sqrt(Br**2 + Bz**2 + Bt**2)
    return Br / norm, Bz / norm


def _parallel_gradient(gradient, Br, Bz, Bt):
    br, bz = _field_aligned_components(Br, Bz, Bt)
    return gradient[:, 0] * br + gradient[:, 1] * bz


def _flatten_solution_with_fields(solutions, Br, Bz, Bt):
    sol, shape = _flatten_solution_nodes(solutions)
    Br_flat, Bz_flat, Bt_flat, _ = _flatten_parallel_inputs(Br, Bz, Bt)
    return sol, Br_flat, Bz_flat, Bt_flat, shape


def _flatten_solution_gradient_with_fields(solutions, gradients, Br, Bz, Bt):
    sol, grad, shape = _flatten_solution_gradient_nodes(solutions, gradients)
    Br_flat, Bz_flat, Bt_flat, _ = _flatten_parallel_inputs(Br, Bz, Bt)
    return sol, grad, Br_flat, Bz_flat, Bt_flat, shape


def _flatten_wall_inputs(solutions, gradients, diffusion, Br, Bz, Bt, normals):
    sol, grad, shape = _flatten_solution_gradient_nodes(solutions, gradients)
    Br_flat, Bz_flat, Bt_flat, _ = _flatten_parallel_inputs(Br, Bz, Bt)
    normals_flat, _ = _flatten_vector_field_impl(normals)
    diffusion_flat, _ = _flatten_scalar_field(diffusion)
    return sol, grad, diffusion_flat, Br_flat, Bz_flat, Bt_flat, normals_flat, shape


def _flatten_vector_field_impl(field):
    if field.ndim > 2:
        shape = field.shape[:2]
        return field.reshape(shape[0] * shape[1], field.shape[2]), shape
    return field, None


def _wall_parallel_projection(Br, Bz, Bt, normals):
    br, bz = _field_aligned_components(Br, Bz, Bt)
    bn = br * normals[:, 0] + bz * normals[:, 1]
    return br, bz, bn

def calculate_plasma_resistivity_cons(solutions,Mref,mD,n0,L0,t0,ohmic_coef,Zeff):
    """
    calculates plasma resistivity based on conservatives values
    """
    temperature = 2 / 3 / Mref * solutions[..., 3] / solutions[..., 0]
    return Zeff * ohmic_coef * n0 * mD * L0**2 / t0**3 / temperature ** (3 / 2)

def calculate_ohmic_source_cons(solutions,jtor,Mref,mD,n0,L0,t0,ohmic_coef,Zeff):
    """
    calculates phmic heating source resistivity based on conservatives values
    """


    eta = calculate_plasma_resistivity_cons(solutions,Mref,mD,n0,L0,t0,ohmic_coef,Zeff)
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
    return solutions[..., cons_idx[b'rho']] * n0

def calculate_grad_n_cons(gradients,n0,L0,cons_idx):
    """
    calculates gradient of plasma density value based on conservatives values
    """
    grad, shape = _flatten_gradient_nodes(gradients)
    res = _grad(grad, b'rho', cons_idx) * n0 / L0
    return _reshape_vector(res, shape)

def calculate_nn_cons(solutions,n0,cons_idx):
    """
    calculates plasma density value based on conservatives values
    """
    return solutions[..., cons_idx[b'rhon']] * n0

def calculate_grad_nn_cons(gradients,n0,L0,cons_idx):
    """
    calculates gradient of neutral density value based on conservatives values
    """
    grad, shape = _flatten_gradient_nodes(gradients)
    res = _grad(grad, b'rhon', cons_idx) * n0 / L0
    return _reshape_vector(res, shape)

def calculate_u_cons(solutions,u0,cons_idx):
    """
    calculates parallel velocity value based on conservatives values
    """
    sol, shape = _flatten_solution_nodes(solutions)
    res = u0 * _sol(sol, b'Gamma', cons_idx) / _sol(sol, b'rho', cons_idx)
    return _reshape_scalar(res, shape)

def calculate_grad_u_cons(solutions,gradients,u0,L0,cons_idx):
    """
    calculates gradient of plasma velocity value based on conservatives values
    """
    sol, grad, shape = _flatten_solution_gradient_nodes(solutions, gradients)
    rho = _sol(sol, b'rho', cons_idx)
    gamma = _sol(sol, b'Gamma', cons_idx)
    res = _ratio_gradient(_grad(grad, b'Gamma', cons_idx), _grad(grad, b'rho', cons_idx), gamma, rho)
    res *= u0 / L0
    return _reshape_vector(res, shape)

def calculate_Ei_cons(solutions,E0,cons_idx):
    """
    calculates ion energy value based on conservatives values
    """
    sol, shape = _flatten_solution_nodes(solutions)
    res = E0 * _sol(sol, b'nEi', cons_idx) / _sol(sol, b'rho', cons_idx)
    return _reshape_scalar(res, shape)

def calculate_grad_Ei_cons(solutions,gradients,E0,L0,cons_idx):
    """
    calculates gradient of ion energy value based on conservatives values
    """
    sol, grad, shape = _flatten_solution_gradient_nodes(solutions, gradients)
    rho = _sol(sol, b'rho', cons_idx)
    nEi = _sol(sol, b'nEi', cons_idx)
    res = _ratio_gradient(_grad(grad, b'nEi', cons_idx), _grad(grad, b'rho', cons_idx), nEi, rho)
    res *= E0 / L0
    return _reshape_vector(res, shape)

def calculate_Ee_cons(solutions,E0,cons_idx):
    """
    calculates electron energy value based on conservatives values
    """
    sol, shape = _flatten_solution_nodes(solutions)
    res = E0 * _sol(sol, b'nEe', cons_idx) / _sol(sol, b'rho', cons_idx)
    return _reshape_scalar(res, shape)

def calculate_grad_Ee_cons(solutions,gradients,E0,L0,cons_idx):
    """
    calculates gradient of electron energy value based on conservatives values
    """
    sol, grad, shape = _flatten_solution_gradient_nodes(solutions, gradients)
    rho = _sol(sol, b'rho', cons_idx)
    nEe = _sol(sol, b'nEe', cons_idx)
    res = _ratio_gradient(_grad(grad, b'nEe', cons_idx), _grad(grad, b'rho', cons_idx), nEe, rho)
    res *= E0 / L0
    return _reshape_vector(res, shape)

def calculate_pi_cons(solutions,p0,cons_idx):
    """
    calculates ion pressure value based on conservatives values
    """
    sol, shape = _flatten_solution_nodes(solutions)
    rho = _sol(sol, b'rho', cons_idx)
    gamma = _sol(sol, b'Gamma', cons_idx)
    res = p0 * (_sol(sol, b'nEi', cons_idx) - 0.5 * gamma**2 / rho)
    return _reshape_scalar(res, shape)

def calculate_grad_pi_cons(solutions,gradients,p0,L0,cons_idx):
    """
    calculates gradient of ion pressure value based on conservatives values
    """
    sol, grad, shape = _flatten_solution_gradient_nodes(solutions, gradients)
    rho = _sol(sol, b'rho', cons_idx)
    gamma = _sol(sol, b'Gamma', cons_idx)
    res = _grad(grad, b'nEi', cons_idx).copy()
    res -= _grad(grad, b'Gamma', cons_idx) * (gamma / rho)[:, None]
    res += 0.5 * _grad(grad, b'rho', cons_idx) * (gamma**2 / rho**2)[:, None]
    res *= p0 / L0
    return _reshape_vector(res, shape)

def calculate_pe_cons(solutions,p0,cons_idx):
    """
    calculates electron pressure value based on conservatives values
    """
    sol, shape = _flatten_solution_nodes(solutions)
    res = p0 * _sol(sol, b'nEe', cons_idx)
    return _reshape_scalar(res, shape)

def calculate_grad_pe_cons(gradients,p0,L0,cons_idx):
    """
    calculates gradient of electron pressure value based on conservatives values
    """
    grad, shape = _flatten_gradient_nodes(gradients)
    res = _grad(grad, b'nEe', cons_idx) * p0 / L0
    return _reshape_vector(res, shape)

def calculate_pdyn_cons(solutions,p0,E0,cons_idx):
    """
    calculates dynamic pressure value based on conservatives values
    kb(Ti+Te)+mD*u**2, first is dimensionalized by p0, second by m0*n0*u0**2
    """
    sol, shape = _flatten_solution_nodes(solutions)
    pe = calculate_pe_cons(sol,1,cons_idx)
    pi = calculate_pi_cons(sol,1,cons_idx)
    u = calculate_u_cons(sol,1,cons_idx)
    n = calculate_n_cons(sol,1,cons_idx)
    res = p0*(pe+pi)+E0*n*u**2
    return _reshape_scalar(res, shape)

def calculate_Ti_cons(solutions,T0,Mref,cons_idx):
    """
    calculates ion temperature value based on conservatives values

    """
    sol, shape = _flatten_solution_nodes(solutions)
    rho = _sol(sol, b'rho', cons_idx)
    gamma = _sol(sol, b'Gamma', cons_idx)
    res = 2 / 3 / Mref * T0 * (_sol(sol, b'nEi', cons_idx) / rho - 0.5 * gamma**2 / rho**2)
    return _reshape_scalar(res, shape)

def calculate_grad_Ti_cons(solutions,gradients,T0,Mref,L0,cons_idx):
    """
    calculates gradient of ion temerature value based on conservatives values
    """
    sol, grad, shape = _flatten_solution_gradient_nodes(solutions, gradients)
    rho = _sol(sol, b'rho', cons_idx)
    gamma = _sol(sol, b'Gamma', cons_idx)
    nEi = _sol(sol, b'nEi', cons_idx)

    res = _grad(grad, b'nEi', cons_idx) / rho[:, None]
    res += _grad(grad, b'rho', cons_idx) * ((gamma**2 / rho**3) - (nEi / rho**2))[:, None]
    res -= _grad(grad, b'Gamma', cons_idx) * (gamma / rho**2)[:, None]
    res *= 2 / 3 / Mref * T0 / L0
    return _reshape_vector(res, shape)

def calculate_grad_Ti_par_cons(solutions,gradients,Br,Bz,Bt,T0,Mref,L0,cons_idx):
    """
    calculates parallel gradient of ion temerature value based on conservatives values
    """
    grad, shape = _flatten_gradient_nodes(gradients)
    Br_res, Bz_res, Bt_res, _ = _flatten_parallel_inputs(Br, Bz, Bt)
    grad_ti = calculate_grad_Ti_cons(solutions, grad, T0, Mref, L0, cons_idx)
    res = _parallel_gradient(grad_ti, Br_res, Bz_res, Bt_res)
    return _reshape_scalar(res, shape)

def calculate_Te_cons(solutions,T0,Mref,cons_idx):
    """
    calculates electron temperature value based on conservatives values
    """
    sol, shape = _flatten_solution_nodes(solutions)
    res = 2 / 3 / Mref * T0 * (_sol(sol, b'nEe', cons_idx) / _sol(sol, b'rho', cons_idx))
    return _reshape_scalar(res, shape)

def calculate_grad_Te_cons(solutions,gradients,T0,Mref,L0,cons_idx):
    """
    calculates gradient of electron temerature value based on conservatives values
    """
    sol, grad, shape = _flatten_solution_gradient_nodes(solutions, gradients)
    rho = _sol(sol, b'rho', cons_idx)
    nEe = _sol(sol, b'nEe', cons_idx)
    res = _ratio_gradient(_grad(grad, b'nEe', cons_idx), _grad(grad, b'rho', cons_idx), nEe, rho)
    res *= T0 / L0 * 2 / 3 / Mref
    return _reshape_vector(res, shape)

def calculate_grad_Te_par_cons(solutions,gradients,Br,Bz,Bt,T0,Mref,L0,cons_idx):
    """
    calculates parallel gradient of electron temerature value based on conservatives values
    """
    grad, shape = _flatten_gradient_nodes(gradients)
    Br_res, Bz_res, Bt_res, _ = _flatten_parallel_inputs(Br, Bz, Bt)
    grad_te = calculate_grad_Te_cons(solutions, grad, T0, Mref, L0, cons_idx)
    res = _parallel_gradient(grad_te, Br_res, Bz_res, Bt_res)
    return _reshape_scalar(res, shape)

def calculate_cs_cons(solutions,u0,cons_idx):
    """
    calculates sound speed of the plasma value based on conservatives values
    """
    sol, shape = _flatten_solution_nodes(solutions)
    rho = _sol(sol, b'rho', cons_idx)
    gamma = _sol(sol, b'Gamma', cons_idx)
    thermal = _sol(sol, b'nEi', cons_idx) + _sol(sol, b'nEe', cons_idx) - 0.5 * gamma**2 / rho
    res = u0 * np.sqrt(2 / 3 * thermal / rho)
    return _reshape_scalar(res, shape)

def calculate_grad_cs_cons(solutions,gradients,u0,L0,cons_idx):
    """
    calculates gradient of sound speed value based on conservatives values
    cs = u0*(2/3*(U3+U4-1/2*U2**2/U1)/U1)**0.5
    grad(cs) = u0/L0/2/(cs/u0)*(2/3)*(grad(U1)*(-U3/U1**2-U4/U1**2+U2**2/U1**3)+
                                      grad(U2)*(-U2/U1**2)+grad(U3)/U1+grad(U4)/U1)
    """
    sol, grad, shape = _flatten_solution_gradient_nodes(solutions, gradients)
    rho = _sol(sol, b'rho', cons_idx)
    gamma = _sol(sol, b'Gamma', cons_idx)
    nEi = _sol(sol, b'nEi', cons_idx)
    nEe = _sol(sol, b'nEe', cons_idx)
    cs = calculate_cs_cons(sol, u0, cons_idx) / u0

    res = _grad(grad, b'rho', cons_idx) * ((gamma**2 / rho - (nEe + nEi)) / rho**2)[:, None]
    res -= _grad(grad, b'Gamma', cons_idx) * (gamma / rho**2)[:, None]
    res += (_grad(grad, b'nEi', cons_idx) + _grad(grad, b'nEe', cons_idx)) / rho[:, None]
    res *= (1 / 3 / cs[:, None])
    res *= u0 / L0
    return _reshape_vector(res, shape)

def calculate_M_cons(solutions,cons_idx):
    """
    calculates Mach number value based on conservatives values
    """
    sol, shape = _flatten_solution_nodes(solutions)
    u = calculate_u_cons(sol,1,cons_idx)
    cs = calculate_cs_cons(sol,1,cons_idx)
    res = u/cs
    return _reshape_scalar(res, shape)

def calculate_grad_M_cons(solutions,gradients,L0,cons_idx):
    """
    calculates gradient of Mach number value based on conservatives values

    """
    sol, grad, shape = _flatten_solution_gradient_nodes(solutions, gradients)
    cs = calculate_cs_cons(sol,1,cons_idx)[:,None]
    grad_cs = calculate_grad_cs_cons(sol,grad,1,1,cons_idx)
    u = calculate_u_cons(sol,1,cons_idx)[:,None]
    grad_u = calculate_grad_u_cons(sol,grad,1,1,cons_idx)

    res = grad_u/cs-grad_cs*u/cs**2
    res /= L0
    return _reshape_vector(res, shape)

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
        res = res.reshape(dimensions[0],dimensions[1])
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
        res = res.reshape(dimensions[0],dimensions[1],dimensions[3])
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
        R_res = R
        q_res = q_cyl

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
    q_cyl = np.minimum(1e4,np.maximum(1,q_cyl))

    
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
    sol, shape = _flatten_solution_nodes(solutions)
    res = gamma0 * _sol(sol, b'Gamma', cons_idx)
    return _reshape_scalar(res, shape)

def calculate_parallel_ion_heat_flux_par_conv_cons(solutions,n0,T0,Mref,kb,mD,u0,cons_idx):
    """
    calculates parallel convective ion heat flux value based on conservatives values
    q_ipar = (5/2*kb*n*Ti+1/2*mD*n*u**2)u
    """
    sol, shape = _flatten_solution_nodes(solutions)
    u = calculate_u_cons(sol,u0,cons_idx)
    ti = calculate_Ti_cons(sol,T0,Mref,cons_idx)
    n = calculate_n_cons(sol,n0,cons_idx)
    res = n*u*(5/2*kb*ti+0.5*mD*u**2)
    return _reshape_scalar(res, shape)

def calculate_parallel_ion_heat_flux_par_cond_cons(solutions,gradients,Br,Bz,Bt,q0,T0,Mref,L0,Tmax,cons_idx):
    """
    calculates parallel conductive ion heat flux value based on conservatives values
    q_ipar = - kappa_par_i*Ti**(5/2)*dTi/dl = q0*Ti**(5/2)*dTi/dl
    """
    sol, grad, Br_res, Bz_res, Bt_res, shape = _flatten_solution_gradient_with_fields(
        solutions, gradients, Br, Bz, Bt
    )
    ti = calculate_Ti_cons(sol,T0,Mref,cons_idx)
    ti = np.minimum(Tmax,ti)
    grad_ti_par = calculate_grad_Ti_par_cons(sol,grad,Br_res,Bz_res,Bt_res,T0,Mref,L0,cons_idx)
    res = -q0*ti**(5/2)*grad_ti_par
    return _reshape_scalar(res, shape)


def calculate_parallel_ion_heat_flux_par_cons(solutions,gradients,Br,Bz,Bt,n0,q0,T0,Mref,kb,mD,u0,L0,Tmax,cons_idx):
    """
    calculates parallel conductive ion heat flux value based on conservatives values
    q_ipar = q_parcond+q_iparconv
    """
    sol, grad, Br_res, Bz_res, Bt_res, shape = _flatten_solution_gradient_with_fields(
        solutions, gradients, Br, Bz, Bt
    )
    q_iconv = calculate_parallel_ion_heat_flux_par_conv_cons(sol,n0,T0,Mref,kb,mD,u0,cons_idx)
    q_icond = calculate_parallel_ion_heat_flux_par_cond_cons(sol,grad,Br_res,Bz_res,Bt_res,q0,T0,Mref,L0,Tmax,cons_idx)

    res = q_iconv+q_icond
    return _reshape_scalar(res, shape)


def calculate_parallel_electron_heat_flux_par_conv_cons(solutions,n0,T0,Mref,kb,u0,cons_idx):
    """
    calculates parallel convective electron heat flux value based on conservatives values
    q_epar = (5/2*kb*n*Te)u
    """
    sol, shape = _flatten_solution_nodes(solutions)
    u = calculate_u_cons(sol,u0,cons_idx)
    te= calculate_Te_cons(sol,T0,Mref,cons_idx)
    n = calculate_n_cons(sol,n0,cons_idx)
    res = n*u*(5/2*kb*te)
    return _reshape_scalar(res, shape)

def calculate_parallel_electron_heat_flux_par_cond_cons(solutions,gradients,Br,Bz,Bt,q0,T0,Mref,L0,Tmax,cons_idx):
    """
    calculates parallel conductive electron heat flux value based on conservatives values
    q_epar = - kappa_par_e*Te**(5/2)*dTe/dl= q0*Te**(5/2)*dTe/dl
    """
    sol, grad, Br_res, Bz_res, Bt_res, shape = _flatten_solution_gradient_with_fields(
        solutions, gradients, Br, Bz, Bt
    )
    te = calculate_Te_cons(sol,T0,Mref,cons_idx)
    te = np.minimum(Tmax,te)
    grad_te_par = calculate_grad_Te_par_cons(sol,grad,Br_res,Bz_res,Bt_res,T0,Mref,L0,cons_idx)
    res = -q0*te**(5/2)*grad_te_par
    return _reshape_scalar(res, shape)


def calculate_parallel_electron_heat_flux_par_cons(solutions,gradients,Br,Bz,Bt,n0,q0,T0,Mref,kb,u0,L0,Tmax,cons_idx):
    """
    calculates parallel conductive electron heat flux value based on conservatives values
    q_epar = q_parcond+q_iparconv
    """
    sol, grad, Br_res, Bz_res, Bt_res, shape = _flatten_solution_gradient_with_fields(
        solutions, gradients, Br, Bz, Bt
    )
    q_econv = calculate_parallel_electron_heat_flux_par_conv_cons(sol,n0,T0,Mref,kb,u0,cons_idx)
    q_econd = calculate_parallel_electron_heat_flux_par_cond_cons(sol,grad,Br_res,Bz_res,Bt_res,q0,T0,Mref,L0,Tmax,cons_idx)

    res = q_econv+q_econd
    return _reshape_scalar(res, shape)

def calculate_particle_perp_flux_wall_cons(solutions,gradients,diffusion,Br,Bz,Bt,n,n0,L0,cons_idx):
    """
    calculates perpendicular particle flux on the wall with normal n value based on conservatives values
    diffusion dimensional (assuming Dperp only)
    """
    _, grad, diffusion_res, Br_res, Bz_res, Bt_res, n_res, shape = _flatten_wall_inputs(
        solutions, gradients, diffusion, Br, Bz, Bt, n
    )
    grad_n = calculate_grad_n_cons(grad,n0,L0,cons_idx)
    br, bz, bn = _wall_parallel_projection(Br_res, Bz_res, Bt_res, n_res)

    res = -1*diffusion_res*(grad_n[:,0]*n_res[:,0]+grad_n[:,1]*n_res[:,1]-grad_n[:,0]*bn*br-grad_n[:,1]*bn*bz)
    return _reshape_scalar(res, shape)

def calculate_perp_ion_heat_wall_cons(solutions,gradients,diffusion,Br,Bz,Bt,n,n0,E0,L0,cons_idx):
    """
    calculates perpendicular ion heat flux on the wall with normal n value based on conservatives values
    diffusion dimensional (assuming Dperp only)
    """
    _, grad, diffusion_res, Br_res, Bz_res, Bt_res, n_res, shape = _flatten_wall_inputs(
        solutions, gradients, diffusion, Br, Bz, Bt, n
    )
    br, bz, bn = _wall_parallel_projection(Br_res, Bz_res, Bt_res, n_res)
    grad_nEi = _grad(grad, b'nEi', cons_idx)

    res = -1*diffusion_res*(grad_nEi[:,0]*n_res[:,0]+grad_nEi[:,1]*n_res[:,1]-grad_nEi[:,0]*bn*br-grad_nEi[:,1]*bn*bz)*n0*E0/L0
    return _reshape_scalar(res, shape)

def calculate_perp_electron_heat_wall_cons(solutions,gradients,diffusion,Br,Bz,Bt,n,n0,E0,L0,cons_idx):
    """
    calculates perpendicular ion heat flux on the wall with normal n value based on conservatives values
    diffusion dimensional (assuming Dperp only)
    """
    _, grad, diffusion_res, Br_res, Bz_res, Bt_res, n_res, shape = _flatten_wall_inputs(
        solutions, gradients, diffusion, Br, Bz, Bt, n
    )
    br, bz, bn = _wall_parallel_projection(Br_res, Bz_res, Bt_res, n_res)
    grad_nEe = _grad(grad, b'nEe', cons_idx)

    res = -1*diffusion_res*(grad_nEe[:,0]*n_res[:,0]+grad_nEe[:,1]*n_res[:,1]-grad_nEe[:,0]*bn*br-grad_nEe[:,1]*bn*bz)*n0*E0/L0
    return _reshape_scalar(res, shape)

def calculate_ion_heat_flux_wall_bc_cons(solutions,gamma_i,Br,Bz,Bt,n,n0,u0,T0,Mref,kb,mD,cons_idx):
    """
    calculates ion heat flux on the wall with normal n value based on conservatives values
    diffusion dimensional (assuming Dperp only)
    """
    sol, Br_res, Bz_res, Bt_res, shape = _flatten_solution_with_fields(solutions, Br, Bz, Bt)
    n_res, _ = _flatten_vector_field_impl(n)
    ni = calculate_n_cons(sol,n0,cons_idx)
    u = calculate_u_cons(sol,u0,cons_idx)
    ti = calculate_Ti_cons(sol,T0,Mref,cons_idx)
    _, _, bn = _wall_parallel_projection(Br_res, Bz_res, Bt_res, n_res)

    res = (gamma_i*u*ni*ti*kb+0.5*ni*mD*u**3)*bn
    return _reshape_scalar(res, shape)

def calculate_electron_heat_flux_wall_bc_cons(solutions,gamma_e,Br,Bz,Bt,n,n0,u0,T0,Mref,kb,cons_idx):
    """
    calculates electron heat flux on the wall with normal n value based on conservatives values
    diffusion dimensional (assuming Dperp only)
    """
    sol, Br_res, Bz_res, Bt_res, shape = _flatten_solution_with_fields(solutions, Br, Bz, Bt)
    n_res, _ = _flatten_vector_field_impl(n)
    ne = calculate_n_cons(sol,n0,cons_idx)
    u = calculate_u_cons(sol,u0,cons_idx)
    te = calculate_Te_cons(sol,T0,Mref,cons_idx)
    _, _, bn = _wall_parallel_projection(Br_res, Bz_res, Bt_res, n_res)

    res = gamma_e*u*ne*te*bn*kb
    return _reshape_scalar(res, shape)
