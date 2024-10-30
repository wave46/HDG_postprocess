import numpy as np
from .tools import softplus,double_softplus
from .atomic import calculate_cx_rate,calculate_cx_rate_cons,calculate_iz_rate,calculate_iz_rate_cons
def calculate_dnn(ti,te,ne,dnn_params,atomic_parameters,kb,mD):
    """
    calculates neutral diffusion value
    """
    if dnn_params['const']:
        return dnn_params['dnn_max']*np.ones_like(ti)
    else:
        if dnn_params['ti_soft']:
            ti = softplus(ti,dnn_params['ti_min'],dnn_params['ti_w'],dnn_params['ti_width'])
        else:
            ti[ti<dnn_params['ti_min']] = dnn_params['ti_min']
        
        sigma_cx = calculate_cx_rate(te,atomic_parameters['cx'])
        sigma_iz = calculate_iz_rate(te,ne,atomic_parameters['iz'])
        dnn = kb*ti/mD/ne/(sigma_cx+sigma_iz)
        if dnn_params['dnn_soft']:
            dnn = double_softplus(dnn, dnn_params['dnn_min'], dnn_params['dnn_max'],
                                  dnn_params['dnn_w'],dnn_params['dnn_width'])
        else:
            dnn[dnn>dnn_params['dnn_max']] = dnn_params['dnn_max']
            dnn[dnn<dnn_params['dnn_min']] = dnn_params['dnn_min']
        
        return dnn

def calculate_dnn_cons(solutions,dnn_params,atomic_parameters,kb,mD,T0,n0,Mref,L0,t0):
    """
    calculates neutral diffusion value based on conservatives values
    """
    dimensions = None
    if dnn_params['const']:
        if len(solutions.shape)>2:
            return dnn_params['dnn_max']*np.ones((solutions.shape[0],solutions.shape[1]))
        else:
            return dnn_params['dnn_max']*np.ones((solutions.shape[0]))
    else:
        if len(solutions.shape)>2:
            dimensions = solutions.shape
            ti = T0*2/3/Mref*(solutions[:,:,2]/solutions[:,:,0]-1/2*solutions[:,:,1]**2/solutions[:,:,0]**2)
            ti = ti.flatten()
            ne = n0*solutions[:,:,0].flatten()
        else:
            ti = T0*2/3/Mref*(solutions[:,2]/solutions[:,0]-1/2*solutions[:,1]**2/solutions[:,0]**2)
            ne = n0*solutions[:,0]
        if dnn_params['ti_soft']:
            ti = softplus(ti,dnn_params['ti_min'],dnn_params['ti_w'],dnn_params['ti_width'])
        else:
            ti[ti<dnn_params['ti_min']] = dnn_params['ti_min']
        
        sigma_cx = calculate_cx_rate_cons(solutions,atomic_parameters['cx'],T0,Mref)
        sigma_iz = calculate_iz_rate_cons(solutions,atomic_parameters['iz'],T0,n0,Mref)
        if dimensions is not None:
            sigma_cx = sigma_cx.flatten()
            sigma_iz = sigma_iz.flatten()

        dnn = kb*ti/mD/ne/(sigma_cx+sigma_iz)
        dnn /= (L0**2/t0)
        if dnn_params['dnn_soft']:
            dnn = double_softplus(dnn, dnn_params['dnn_min_adim'], dnn_params['dnn_max_adim'],
                                  dnn_params['dnn_w'],dnn_params['dnn_width'])
        else:
            dnn[dnn>dnn_params['dnn_max_adim']] = dnn_params['dnn_max_adim']
            dnn[dnn<dnn_params['dnn_min_adim']] = dnn_params['dnn_min_adim']
        
        if dimensions is not None:
            dnn = dnn.reshape(dimensions[0],dimensions[1])
        
        return dnn*(L0**2/t0)

def calculate_dnn_with_nn_collision_cons(solutions,dnn_params,atomic_parameters,kb,mD,T0,n0,Mref,L0,t0):

    """

    calculates neutral diffusion value with neutral-neutral collisions based on conservatives values

    """

    dimensions = None
    if dnn_params['const']:
        if len(solutions.shape)>2:
            return dnn_params['dnn_max']*np.ones((solutions.shape[0],solutions.shape[1]))
        else:
            return dnn_params['dnn_max']*np.ones((solutions.shape[0]))
    else:
        if len(solutions.shape)>2:
            dimensions = solutions.shape
            ti = T0*2/3/Mref*(solutions[:,:,2]/solutions[:,:,0]-1/2*solutions[:,:,1]**2/solutions[:,:,0]**2)
            ti = ti.flatten()
            ne = n0*solutions[:,:,0].flatten()
            nn = n0*solutions[:,:,4].flatten()
        else:
            ti = T0*2/3/Mref*(solutions[:,2]/solutions[:,0]-1/2*solutions[:,1]**2/solutions[:,0]**2)
            ne = n0*solutions[:,0]
            nn = n0*solutions[:,4]
        if dnn_params['ti_soft']:
            ti = softplus(ti,dnn_params['ti_min'],dnn_params['ti_w'],dnn_params['ti_width'])
        else:
            ti[ti<dnn_params['ti_min']] = dnn_params['ti_min']
        

        sigma_cx = calculate_cx_rate_cons(solutions,atomic_parameters['cx'],T0,Mref)
        sigma_iz = calculate_iz_rate_cons(solutions,atomic_parameters['iz'],T0,n0,Mref)



        s0 = 5.2958*10**(-11) *10**(-6)      # dimensional, m^3/s

        sigma_nn_collision = s0 * (ti*kb/(1.38064852e-23))**0.25



        if dimensions is not None:
            sigma_cx = sigma_cx.flatten()
            sigma_iz = sigma_iz.flatten()
            sigma_nn_collision = sigma_nn_collision.flatten()


        dnn = kb*ti/mD/( ne*(sigma_cx+sigma_iz) +nn*sigma_nn_collision)
        dnn /= (L0**2/t0)
        if dnn_params['dnn_soft']:
            dnn = double_softplus(dnn, dnn_params['dnn_min_adim'], dnn_params['dnn_max_adim'],
                                  dnn_params['dnn_w'],dnn_params['dnn_width'])

        else:
            dnn[dnn>dnn_params['dnn_max_adim']] = dnn_params['dnn_max_adim']
            dnn[dnn<dnn_params['dnn_min_adim']] = dnn_params['dnn_min_adim']

        if dimensions is not None:
            dnn = dnn.reshape(dimensions[0],dimensions[1])        

        return dnn*(L0**2/t0)

def calculate_mfp(ti,te,ne,dnn_params,atomic_parameters,kb,mD):
    """
    calculates neutral mean free path value based on conservatives values
    """
    dimensions = None
    


    dnn = calculate_dnn(ti,te,ne,dnn_params,atomic_parameters,kb,mD)
    if dnn_params['ti_soft']:
        ti = softplus(ti,dnn_params['ti_min'],dnn_params['ti_w'],dnn_params['ti_width'])
    else:
        ti[ti<dnn_params['ti_min']] = dnn_params['ti_min']
    mfp = 2*dnn/np.sqrt(kb*ti/mD)

    return mfp

def calculate_mfp_cons(solutions,dnn_params,atomic_parameters,kb,mD,T0,n0,Mref,L0,t0):
    """
    calculates neutral mean free path value based on conservatives values
    """
    dimensions = None
    
    if len(solutions.shape)>2:
        dimensions = solutions.shape
        ti = T0*2/3/Mref*(solutions[:,:,2]/solutions[:,:,0]-1/2*solutions[:,:,1]**2/solutions[:,:,0]**2)
        ti = ti.flatten()
        dnn = calculate_dnn_cons(solutions,dnn_params,atomic_parameters,kb,mD,T0,n0,Mref,L0,t0).flatten()
    else:
        ti = T0*2/3/Mref*(solutions[:,2]/solutions[:,0]-1/2*solutions[:,1]**2/solutions[:,0]**2)
        dnn = calculate_dnn_cons(solutions,dnn_params,atomic_parameters,kb,mD,T0,n0,Mref,L0,t0)
    if dnn_params['ti_soft']:
        ti = softplus(ti,dnn_params['ti_min'],dnn_params['ti_w'],dnn_params['ti_width'])
    else:
        ti[ti<dnn_params['ti_min']] = dnn_params['ti_min']
    mfp = 2*dnn/np.sqrt(kb*ti/mD)
    if dimensions is not None:
        mfp = mfp.reshape(dimensions[0],dimensions[1])
    return mfp


