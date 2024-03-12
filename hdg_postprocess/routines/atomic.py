import numpy as np

def d_eirene_fit_dte(M,params):
    """
    partial derivative value in log log space
    """
    t_shape,ne_shape = params.shape
    te,ne = M
    result = 0
    for i in range(1,t_shape):
        for j in range(ne_shape):
            result+=params[i,j]*i*np.log(ne/1e14)**(j)*np.log(te)**(i-1)
    return result

def eirene_fit(M,params,te_min,te_max,ne_min, ne_max):
    te,ne = M
    t_shape,ne_shape = params.shape    
    result = np.zeros_like(te) 
    ## region 1, in the range of applicability
    # part in good ne, te:
    index_1 = (ne>=ne_min)*(ne<=ne_max)*(te>=te_min)*(te<=te_max) 
    for i in range(t_shape):
        for j in range(ne_shape):
            result[index_1]+=params[i,j]*np.log(ne[index_1]/1e14)**j*np.log(te[index_1])**i
    ##region 2
    # ne<ne_min, te in good range:
    index_2 = (ne<ne_min)*(te>=te_min)*(te<=te_max) 
    for i in range(t_shape):
        result[index_2]+=params[i,0]*np.log(te[index_2])**i


    ##region 3 
    #ne<ne_min, te<te_min 
    index_3 = (ne<ne_min)*(te<te_min)
    for i in range(t_shape):
        #for j in range(ne_shape):
        result[index_3]+=params[i,0]*np.log(te_min)**i
    result[index_3] += d_eirene_fit_dte(np.vstack([te_min*np.ones_like(te[index_3]),ne_min*np.ones_like(ne[index_3])]), params)*(np.log(te[index_3])-np.log(te_min))
   
    ##region 4 
    #ne good, te<te_min 
    index_4 = (ne>=ne_min)*(ne<=ne_max)*(te<te_min)
    for i in range(t_shape):
        for j in range(ne_shape):
            result[index_4]+=params[i,j]*np.log(ne[index_4]/1e14)**j*np.log(te_min)**i
    result[index_4] += d_eirene_fit_dte(np.vstack([te_min*np.ones_like(te[index_4]),ne[index_4]]), params)*(np.log(te[index_4])-np.log(te_min))

    ##region 5 
    #ne>ne_max, te<te_min 
    index_5 = (ne>ne_max)*(te<te_min)
    for i in range(t_shape):
        for j in range(ne_shape):
            result[index_5]+=params[i,j]*np.log(ne_max/1e14)**j*np.log(te_min)**i
    result[index_5] += d_eirene_fit_dte(np.vstack([te_min*np.ones_like(te[index_5]),ne_max*np.ones_like(ne[index_5])]), params)*(np.log(te[index_5])-np.log(te_min))

    ##region 6 
    #ne>ne_max, te norm
    index_6 = (ne>ne_max)*(te>=te_min)*(te<=te_max) 
    for i in range(t_shape):
        for j in range(ne_shape):
            result[index_6]+=params[i,j]*np.log(ne_max/1e14)**j*np.log(te[index_6])**i

    ##region 7 
    #ne>ne_max, te>te_max
    index_7 = (ne>ne_max)*(te>te_max)
    for i in range(t_shape):
        for j in range(ne_shape):
            result[index_7]+=params[i,j]*np.log(ne_max/1e14)**j*np.log(te_max)**i
    result[index_7] += d_eirene_fit_dte(np.vstack([te_max*np.ones_like(te[index_7]),ne_max*np.ones_like(ne[index_7])]), params)*(np.log(te[index_7])-np.log(te_max))        
    

    ##region 8 
    #ne normal, te>te_max
    index_8 = (ne>=ne_min)*(ne<=ne_max)*(te>te_max)
    for i in range(t_shape):
        for j in range(ne_shape):
            result[index_8]+=params[i,j]*np.log(ne[index_8]/1e14)**j*np.log(te_max)**i
    result[index_8] += d_eirene_fit_dte(np.vstack([te_max*np.ones_like(te[index_8]),ne[index_8]]), params)*(np.log(te[index_8])-np.log(te_max)) 

    ##region 9 
    #ne<ne_min, te>te_max
    index_9 = (ne<ne_min)*(te>te_max)
    for i in range(t_shape):
        #for j in range(ne_shape):
        result[index_9]+=params[i,0]*np.log(te_max)**i
    result[index_9] += d_eirene_fit_dte(np.vstack([te_max*np.ones_like(te[index_9]),ne_min*np.ones_like(ne[index_9])]), params)*(np.log(te[index_9])-np.log(te_max))    
    


    if (result>1).any():
        print('WARNING')
        print(ne,te)
    result = np.exp(result)/1e6
    return result

def calculate_iz_rate(te,ne,iz_parameters):
    """
    calculates cx rate for given te,ne
    """
    database = iz_parameters['database']
    alpha = iz_parameters['alpha']
    te_min = iz_parameters['te_min']
    te_max = iz_parameters['te_max']
    ne_min = iz_parameters['ne_min']
    ne_max = iz_parameters['ne_max']
    if database == "AMJUEL 2.1.5JH":
        return eirene_fit(np.vstack([te,ne]),alpha,te_min,te_max,ne_min,ne_max)

def calculate_iz_rate_cons(solutions,iz_parameters,T0,n0,Mref,tol=1e-20):
    """
    calculates cx rate for given te,ne
    """
    database = iz_parameters['database']
    alpha = iz_parameters['alpha']
    te_min = iz_parameters['te_min']
    te_max = iz_parameters['te_max']
    ne_min = iz_parameters['ne_min']
    ne_max = iz_parameters['ne_max']
    if database == "AMJUEL 2.1.5JH":
        te = np.zeros_like(solutions[:,0])
        ne = np.zeros_like(solutions[:,0])
        #good U1 and U4
        good_idx = (solutions[:,0]>tol)&(solutions[:,3]>tol)
        te[good_idx] = T0*2/3/Mref*solutions[good_idx,3]/solutions[good_idx,0]
        ne[good_idx] = n0*solutions[good_idx,0]
        te[~good_idx] = 1e-10
        ne[~good_idx] = n0*1e-20
        return eirene_fit(np.vstack([te,ne]),alpha,te_min,te_max,ne_min,ne_max)

def calculate_iz_source(te,ne,nn,iz_parameters):
    """
    calculates ionization source for given plasma density, electron temperature and neutral density
    """

    sigma_iz = calculate_iz_rate(te,ne,iz_parameters)
    return ne*nn*sigma_iz


def calculate_iz_source_cons(solutions,iz_parameters,T0,n0,Mref):
    """
    calculates ionization source for given conservative solutions
    todo make indexing not hardcoded
    """

    sigma_iz = calculate_iz_rate_cons(solutions,iz_parameters,T0,n0,Mref)
    return n0**2*solutions[:,0]*solutions[:,-1]*sigma_iz


