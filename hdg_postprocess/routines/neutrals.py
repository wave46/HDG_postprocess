import numpy as np
from .tools import softplus,double_softplus
from .atomic import calculate_cx_rate,calculate_cx_rate_cons,calculate_iz_rate,calculate_iz_rate_cons
from .plasma import calculate_grad_nn_cons


def _constant_solution_shape(solutions, value):
    shape = solutions.shape[:2] if solutions.ndim > 2 else (solutions.shape[0],)
    return np.full(shape, value)


def _flatten_node_values(values):
    if values.ndim > 2:
        return values.reshape(values.shape[0] * values.shape[1], *values.shape[2:]), values.shape[:2]
    return values, None


def _flatten_grid(values):
    if values.ndim > 1:
        return values.reshape(values.shape[0] * values.shape[1], *values.shape[2:])
    return values


def _reshape_node_values(values, dimensions):
    return values.reshape(dimensions) if dimensions is not None else values


def _ion_temperature_from_solutions(solutions, T0, Mref):
    return T0 * 2 / 3 / Mref * (solutions[:, 2] / solutions[:, 0] - 0.5 * solutions[:, 1] ** 2 / solutions[:, 0] ** 2)


def _apply_ti_floor(ti, dnn_params):
    if dnn_params['ti_soft']:
        return softplus(ti, dnn_params['ti_min'], dnn_params['ti_w'], dnn_params['ti_width'])
    return np.maximum(ti, dnn_params['ti_min'])


def _apply_dnn_limits(dnn, dnn_params, adimensional=False):
    if adimensional:
        dnn_min = dnn_params['dnn_min_adim']
        dnn_max = dnn_params['dnn_max_adim']
    else:
        dnn_min = dnn_params['dnn_min']
        dnn_max = dnn_params['dnn_max']

    if dnn_params['dnn_soft']:
        return double_softplus(dnn, dnn_min, dnn_max, dnn_params['dnn_w'], dnn_params['dnn_width'])
    return np.clip(dnn, dnn_min, dnn_max)


def calculate_dnn(ti,te,ne,dnn_params,atomic_parameters,kb,mD):
    """
    calculates neutral diffusion value
    """
    if dnn_params['const']:
        return dnn_params['dnn_max'] * np.ones_like(ti)

    ti = _apply_ti_floor(ti, dnn_params)
    sigma_cx = calculate_cx_rate(te,atomic_parameters['cx'])
    sigma_iz = calculate_iz_rate(te,ne,atomic_parameters['iz'])
    dnn = kb * ti / mD / ne / (sigma_cx + sigma_iz)
    return _apply_dnn_limits(dnn, dnn_params)

def calculate_dnn_cons(solutions,dnn_params,atomic_parameters,kb,mD,T0,n0,Mref,L0,t0):
    """
    calculates neutral diffusion value based on conservatives values
    """
    if dnn_params['const']:
        return _constant_solution_shape(solutions, dnn_params['dnn_max'])

    flat_solutions, dimensions = _flatten_node_values(solutions)
    ti = _apply_ti_floor(_ion_temperature_from_solutions(flat_solutions, T0, Mref), dnn_params)
    ne = n0 * flat_solutions[:, 0]
    sigma_cx = calculate_cx_rate_cons(flat_solutions, atomic_parameters['cx'], T0, Mref)
    sigma_iz = calculate_iz_rate_cons(flat_solutions, atomic_parameters['iz'], T0, n0, Mref)

    dnn = kb * ti / mD / ne / (sigma_cx + sigma_iz)
    dnn = _apply_dnn_limits(dnn / (L0**2 / t0), dnn_params, adimensional=True)
    return _reshape_node_values(dnn * (L0**2 / t0), dimensions)

def calculate_dnn_with_nn_collision_cons(solutions,dnn_params,atomic_parameters,kb,mD,T0,n0,Mref,L0,t0):

    """

    calculates neutral diffusion value with neutral-neutral collisions based on conservatives values

    """

    if dnn_params['const']:
        return _constant_solution_shape(solutions, dnn_params['dnn_max'])

    flat_solutions, dimensions = _flatten_node_values(solutions)
    ti = _apply_ti_floor(_ion_temperature_from_solutions(flat_solutions, T0, Mref), dnn_params)
    ne = n0 * flat_solutions[:, 0]
    nn = n0 * flat_solutions[:, 4]

    sigma_cx = calculate_cx_rate_cons(flat_solutions, atomic_parameters['cx'], T0, Mref)
    sigma_iz = calculate_iz_rate_cons(flat_solutions, atomic_parameters['iz'], T0, n0, Mref)
    s0 = 5.2958 * 10 ** (-11) * 10 ** (-6)
    sigma_nn_collision = s0 * (ti * kb / 1.38064852e-23) ** 0.25

    dnn = kb * ti / mD / (ne * (sigma_cx + sigma_iz) + nn * sigma_nn_collision)
    dnn = _apply_dnn_limits(dnn / (L0**2 / t0), dnn_params, adimensional=True)
    return _reshape_node_values(dnn * (L0**2 / t0), dimensions)
    
def calculate_neutral_perp_flux_wall_cons(solutions,gradients,dnn_parameters,atomic_parameters,Br,Bz,Bt,n,n0,L0,charge,m_i,T0,Mref,t0,cons_idx):
    """
    calculates perpendicular neutral flux on the wall with normal n value based on conservatives values
    diffusion dimensional (assuming Dperp only)
    """
    sol, dimensions = _flatten_node_values(solutions)
    grad, _ = _flatten_node_values(gradients)
    Br_res = _flatten_grid(Br)
    Bz_res = _flatten_grid(Bz)
    Bt_res = _flatten_grid(Bt)
    n_res = _flatten_grid(n)

    grad_nn = calculate_grad_nn_cons(grad,n0,L0,cons_idx)

    br = Br_res/np.sqrt(Br_res**2+Bz_res**2+Bt_res**2)
    bz = Bz_res/np.sqrt(Br_res**2+Bz_res**2+Bt_res**2)
    bn = br*n_res[:,0]+bz*n_res[:,1]
    diffusion_res = calculate_dnn_with_nn_collision_cons(sol,dnn_parameters,atomic_parameters,
                                                        charge,m_i,T0,n0,Mref,L0,t0)

    res = diffusion_res*(grad_nn[:,0]*n_res[:,0]+grad_nn[:,1]*n_res[:,1]-grad_nn[:,0]*bn*br-grad_nn[:,1]*bn*bz)

    return _reshape_node_values(res, dimensions)

def calculate_mfp(ti,te,ne,dnn_params,atomic_parameters,kb,mD):
    """
    calculates neutral mean free path value based on conservatives values
    """
    dnn = calculate_dnn(ti,te,ne,dnn_params,atomic_parameters,kb,mD)
    ti = _apply_ti_floor(ti, dnn_params)
    mfp = 2*dnn/np.sqrt(kb*ti/mD)

    return mfp

def calculate_mfp_cons(solutions,dnn_params,atomic_parameters,kb,mD,T0,n0,Mref,L0,t0):
    """
    calculates neutral mean free path value based on conservatives values
    """
    flat_solutions, dimensions = _flatten_node_values(solutions)
    ti = _apply_ti_floor(_ion_temperature_from_solutions(flat_solutions, T0, Mref), dnn_params)
    dnn = calculate_dnn_cons(solutions,dnn_params,atomic_parameters,kb,mD,T0,n0,Mref,L0,t0)
    dnn = _flatten_grid(dnn)
    mfp = 2*dnn/np.sqrt(kb*ti/mD)
    return _reshape_node_values(mfp, dimensions)
