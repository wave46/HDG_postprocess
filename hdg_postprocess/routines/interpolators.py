import numpy as np

class SoledgeHDG2DInterpolator():



    def __init__(self,vertex_coords,vertex_data,connectivity,element_number,
                    reference_element_coordinates,element_type,p_order,limit=True,default_value=0):
        """
        in case of triangles reference element coordinates in 2D
        in case of quadrilaterals reference element coordinates in 1D
        """

        self._connectivity = connectivity
        self._vertex_data = vertex_data
        self._vertex_coords = vertex_coords
        self._element_number = element_number
        self._reference_element_coordinates = reference_element_coordinates
        self._vandermonde = Vandermonde_LP (p_order, reference_element_coordinates)
        self._inv_vandermonde = np.linalg.inv(self._vandermonde)
        self._limit = limit
        self._default_value =  default_value
        self._p_order = p_order
        self._element_type = element_type
        #self._hashed_points = []
        self._hashed_shape_functions = {}
        self._hashed_element = {}


    def evaluate(self,x,y):
        result = 0

        if (x,y) in self._hashed_shape_functions.keys():
            if self._hashed_element[(x,y)]==-1:
                return self._default_value
            else: 
                shape_functions = self._hashed_shape_functions[(x,y)]
                element_data = self._vertex_data[self._hashed_element[(x,y)],:]
                result = np.dot(shape_functions,element_data)
            return result
        else:
            element_number = int(self._element_number(x,y))
            self._hashed_element[(x,y)] = element_number
            if (element_number==-1):
                self._hashed_shape_functions[(x,y)] = [0]              
                if self._limit:            
                    raise ValueError("Requested value outside mesh bounds.")
                else:
                    return self._default_value
            else:
                # get element vertces coordinates
                element_vertices = self._vertex_coords[self._connectivity[element_number,:]]
                #transition to element local coordinates
                xieta = xieta_element_precise(x, y, element_vertices,self._p_order,self._inv_vandermonde,self._element_type)
                #calculating shape functions
                if self._element_type == 'triangle':
                    shape_functions = orthopoly2D(xieta[0], xieta[1], self._p_order) @ self._inv_vandermonde
                elif self._element_type == 'quadrilateral':
                    shape_functions = shapefunctions_quads(xieta[0], xieta[1], self._p_order,self._inv_vandermonde)
                    shape_functions = shape_functions[:,0]
                self._hashed_shape_functions[(x,y)] =shape_functions
                #get data in element vertices
                element_data = self._vertex_data[element_number,:]

                #getting value in point with shape functiosn
                result = np.dot(shape_functions,element_data)

                return result
            
    def __getstate__(self):
        return self._vertex_data, self._element_number, self._limit, self._default_value
    
    def __setstate__(self, state):
        self._vertex_data, self._element_number, self._limit, self._default_value = state

    def __reduce__(self):
        return self.__new__, (self.__class__, ), self.__getstate__()

    def __call__(self,x,y):
        """
        Calculates interpolateion in given point (R,Z)
        """
        return self.evaluate(x,y)

    @classmethod
    def instance(cls,instance,vertex_data=None,limit=None,default_value=None):
        m = SoledgeHDG2DInterpolator.__new__(SoledgeHDG2DInterpolator)
        m._element_number = instance._element_number
        m._connectivity = instance._connectivity
        m._vertex_coords = instance._vertex_coords
        m._reference_element_coordinates = instance._reference_element_coordinates
        m._vandermonde = instance._vandermonde
        m._inv_vandermonde = instance._inv_vandermonde
        m._p_order = instance._p_order
        m._element_type = instance._element_type

        m._hashed_shape_functions = instance._hashed_shape_functions
        m._hashed_element = instance._hashed_element

        # do we have replacement vertex data?
        if vertex_data is None:
            m._vertex_data = instance._vertex_data
        else:
            m._vertex_data = np.array(vertex_data, dtype=np.float64).reshape((m._connectivity.shape[0],m._connectivity.shape[1]))
            if m._vertex_data.ndim != 2 or m._vertex_data.shape[0] != instance._vertex_data.shape[0]:
                raise ValueError("Vertex_data dimensions are incompatible with the number of vertices in the instance ({} vertices).".format(instance._vertex_data.shape[0]))
        
         # do we have a replacement limit check setting?
        if limit is None:
            m._limit = instance._limit
        else:
            m._limit = limit

        # do we have a replacement default value?
        if default_value is None:
            m._default_value = instance._default_value
        else:
            m._default_value = default_value
        
        return m

def xieta_element (x,y,vertices_element,eltype):
    # find local element coordinates xi, eta
    # todo simplified so far (straight triangles)
    J = np.zeros([2,2])
    invJ = np.zeros([2,2])
    xieta = np.zeros(2, dtype = np.float64, order = 'C')

    if eltype == 'triangle':
        #compute jacobian
        J[0,0] = (vertices_element[1,0]-vertices_element[0,0])/2
        J[0,1] = (vertices_element[1,1]-vertices_element[0,1])/2  
        J[1,0] = (vertices_element[2,0]-vertices_element[0,0])/2
        J[1,1] = (vertices_element[2,1]-vertices_element[0,1])/2
        auxx = x - (vertices_element[1,0]+vertices_element[2,0])/2
        auxy = y - (vertices_element[1,1]+vertices_element[2,1])/2

        
    elif eltype == 'quadrilateral':
        #compute jacobian
        J[0,0] = vertices_element[1,0]+vertices_element[2,0]-vertices_element[0,0]-vertices_element[3,0]
        J[0,1] = vertices_element[1,1]+vertices_element[2,1]-vertices_element[0,1]-vertices_element[3,1]
        J[1,0] = vertices_element[2,0]+vertices_element[3,0]-vertices_element[1,0]-vertices_element[0,0]
        J[1,1] = vertices_element[2,1]+vertices_element[3,1]-vertices_element[1,1]-vertices_element[0,1]
        J /= 4
        auxx = x - 0.25*(vertices_element[1,0]+vertices_element[2,0]+vertices_element[0,0]+vertices_element[3,0])
        auxy = y - 0.25*(vertices_element[1,1]+vertices_element[2,1]+vertices_element[0,1]+vertices_element[3,1])

    #inverse matrix
    detJ = J[0,0]*J[1,1]-J[0,1]*J[1,0]
    invJ[0,0] = J[1,1]/detJ
    invJ[0,1] = -1*J[1,0]/detJ
    invJ[1,0] = -1*J[0,1]/detJ
    invJ[1,1] = J[0,0]/detJ
    #compute coordinates
    xieta[0] = auxx*invJ[0,0] + auxy*invJ[0,1]
    xieta[1] = auxx*invJ[1,0] + auxy*invJ[1,1]
    return xieta  


def xieta_element_precise (x,y,vertices_element,p_order,inv_vandermonde,eltype):
    # find local element coordinates xi, eta
    # todo simplified so far (straight triangles)
    tol = 1e-8
    maxit = 100
    invJ = np.zeros([2,2])
    J = np.zeros([2,2])
    xieta0 = np.zeros(2, dtype = np.float64, order = 'C')

    #first, try supposing that element is straight

    xieta0 = xieta_element(x,y,vertices_element,eltype)
    # back projection
    if eltype == 'triangle':
        [x_back,y_back] = (orthopoly2D(xieta0[0], xieta0[1], p_order)@inv_vandermonde)@ vertices_element
    elif eltype == 'quadrilateral':
        shape_functions = shapefunctions_quads(xieta0[0], xieta0[1], p_order,inv_vandermonde)
        [x_back,y_back] = shape_functions[:,0]@vertices_element
    if np.sqrt(((x_back-x)**2+(y_back-y)**2))<tol*np.sqrt(x**2+y**2)+1e-14:
        return xieta0
    else:
        for i in range(maxit):
            if np.sqrt(((x_back-x)**2+(y_back-y)**2))<tol*np.sqrt(x**2+y**2)+1e-14:
                return xieta0
            if eltype == 'triangle':
                _,dp_dxi,dp_deta = orthopoly2D_deriv_xieta(xieta0[0], xieta0[1],p_order)
                Nx = (dp_dxi@inv_vandermonde)
                Ny = (dp_deta@inv_vandermonde)
            elif eltype == 'quadrilateral':
                shape_functions = shapefunctions_quads(xieta0[0], xieta0[1], p_order,inv_vandermonde)                
                Nx = shape_functions[:,1]
                Ny = shape_functions[:,2]
            J[0,0] = Nx@vertices_element[:,0]
            J[1,0] = Ny@vertices_element[:,0]
            J[0,1] = Nx@vertices_element[:,1]
            J[1,1] = Ny@vertices_element[:,1]
            # inverse matrix
            detJ = J[0,0]*J[1,1]-J[0,1]*J[1,0]
            invJ[0,0] = J[1,1]/detJ
            invJ[0,1] = -1*J[1,0]/detJ
            invJ[1,0] = -1*J[0,1]/detJ
            invJ[1,1] = J[0,0]/detJ

            rhs = np.array([x-x_back,y-y_back])[None].T
            
            xieta0 = xieta0+((invJ@rhs).T)[0]
            if eltype == 'triangle':
                [x_back,y_back] = (orthopoly2D(xieta0[0], xieta0[1], p_order)@inv_vandermonde)@ vertices_element
            elif eltype == 'quadrilateral':
                shape_functions = shapefunctions_quads(xieta0[0], xieta0[1], p_order,inv_vandermonde)
                [x_back,y_back] = shape_functions[:,0]@vertices_element

        if np.sqrt(((x_back-x)**2+(y_back-y)**2))>(tol*np.sqrt(x**2+y**2)+1e-14):
            raise Warning("Not converging")
        return xieta0
def Vandermonde_LP(p_order,coord):
    # Vandermonde matrix calculation

    nsd = coord.shape[1]
    N = coord.shape[0]
    V = np.zeros([N,N])
    if nsd == 1:
        if N!=p_order+1:
            raise ValueError ('The number of polynomials does not coincide with the number of nodes')
        for i in range(N):
            V[i,:] = orthopoly1D(coord[i], p_order).T
    elif nsd == 2:
        if N!=(p_order+1)*(p_order+2)/2:
            raise ValueError ('The number of polynomials does not coincide with the number of nodes')
        for i in range(N):
            p = orthopoly2D(coord[i,0], coord[i,1], p_order)
            for j in range(N):
                V[i,j] = p[j]
    else:
        raise ValueError ('wrong dimentsion of coord')
    return V

def orthopoly1D(x,n):
    ''' Computes the ortogonal base of 1D polynomials of degree less 
     or equal to n at the point x in [-1,1]
    '''
    p = np.zeros([n+1,x.shape[0]])
    for i in range(n+1):
        factor = np.sqrt((2*i+1)/2)
        p[i,:] = jacobi(i,0,0,x)*factor

    return p

def orthopoly1D_deriv(x,n):
    ''' Computes the ortogonal base of 1D polynomials and their derivatives of degree less 
     or equal to n at the point x in [-1,1]
    '''
    p = np.zeros([n+1,1])
    dp = np.zeros([n+1,1])
    p[0,:] = 1/np.sqrt(2)
    dp[0,:] = 0
    for i in range(1,n+1):        
        factor = np.sqrt((2*i+1)/2)
        p[i,:] = jacobi(i,0,0,x)*factor
        dp[i,:] = jacobi(i-1,1,1,x)*factor*(i+1)/2
    return p,dp

def shapefunctions_quads(xi,eta,p_order,inv_vandermonde):
    ''' Computes the ortogonal base of 2D polynomials of degree less 
     or equal to n at the point x=(xi,eta) in the reference quadrangle
    '''
    # First calculating 1D shape functions
    sf1d = np.zeros((2,inv_vandermonde.shape[0],2))

    p,dp = orthopoly1D_deriv(xi,p_order)
    sf1d[0,:,0] = p.T@inv_vandermonde
    sf1d[0,:,1] = dp.T@inv_vandermonde

    p,dp = orthopoly1D_deriv(eta,p_order)
    sf1d[1,:,0] = p.T@inv_vandermonde
    sf1d[1,:,1] = dp.T@inv_vandermonde
    

    shapefunctions = np.zeros((inv_vandermonde.shape[0]**2,3))
    perm = permutations_quads[p_order]
    aux = ((sf1d[0,:,0][None].T@sf1d[1,:,0][None]).T).flatten()
    auxx = ((sf1d[0,:,1][None].T@sf1d[1,:,0][None]).T).flatten()
    auxy = ((sf1d[0,:,0][None].T@sf1d[1,:,1][None]).T).flatten()
    shapefunctions[perm,0] = aux
    shapefunctions[perm,1] = auxx
    shapefunctions[perm,2] = auxy


    return shapefunctions
    


def orthopoly2D(xi,eta,n):
    ''' Computes the ortogonal base of 2D polynomials of degree less 
     or equal to n at the point x=(xi,eta) in the reference triangle
    '''

    #translate to r, s coordinates
    if eta != 1:
        r = 2*(1+xi)/(1-eta) - 1
        s = eta
    else: 
        r = -1
        s = 1

    p = orthopoly2D_rst(r, s, n)

    return p

def orthopoly2D_rst(r,s,n):
    '''% p = orthopoly2D_rst(x,n)
    Computes the ortogonal base of 2D polynomials of degree less 
    or equal to n at the point x=(r,s) in [-1,1]^2
    '''
    ncount = 0
    N = int((n+1)*(n+2)/2)
    p = np.zeros(N)
    for ndeg in range(n+1):
        for i in range(0,ndeg+1):
            if i == 0:
                p_i = 1
                q_i = 1
            else:
                p_i = jacobi(i,0,0,r)
                q_i = q_i*(1-s)/2
            #Value for j
            j = ndeg-i

            if j == 0:
                p_j = 1
            else:
                p_j = jacobi(j, 2*i+1, 0, s)

            factor = np.sqrt((2*i+1)*(i+j+1)/2)
            
            p[ncount] = ((p_i*q_i*p_j)*factor)
            ncount+=1
    return p

def orthopoly2D_deriv_xieta(xi,eta,n):
    if eta != 1:
        r = 2*(1+xi)/(1-eta) - 1
        s = eta
    else: 
        r = -1
        s = 1
    
    p,dp_dxi,dp_deta = orthopoly2D_deriv_rst(r,s,n)
    return p,dp_dxi,dp_deta

def orthopoly2D_deriv_rst(r,s,n):
    tol = 1e-14
    ncount = 0
    N = int((n+1)*(n+2)/2)
    p = np.zeros(N)
    dp_dxi = np.zeros(N)
    dp_deta = np.zeros(N)
    xi = (1+r)*(1-s)/2-1
    #if (1-s)<tol:
    #    s = 1-tol
    eta = s

    dr_dxi = 2/(1-eta)
    dr_deta = 2*(1+xi)/(1-eta)**2


    for ndeg in range(n+1):
        for i in range(0,ndeg+1):
            if i == 0:
                p_i = 1
                q_i = 1 
                dp_i = 0
                dq_i = 0
            else:
                p_i = jacobi(i,0,0,r)
                dp_i = jacobi(i-1,1,1,r)*(i+1)/2
                q_i = q_i*(1-s)/2
                dq_i = q_i*(-i)/(1-s)
            #Value for j
            j = ndeg-i

            if j == 0:
                p_j = 1
                dp_j = 0
            else:
                p_j = jacobi(j, 2*i+1, 0, s)
                dp_j = jacobi(j-1, 2*i+2, 1, s)*(j+2*i+2)/2
            
            factor = np.sqrt((2*i+1)*(i+j+1)/2)
            p[ncount] = ((p_i*q_i*p_j)*factor)
            dp_dr = (dp_i*q_i*p_j)*factor
            dp_ds = (p_i*(dq_i*p_j+q_i*dp_j))*factor
            dp_dxi[ncount] = dp_dr*dr_dxi
            dp_deta[ncount] = dp_dr*dr_deta+dp_ds
            ncount+=1
    return p,dp_dxi,dp_deta


def  jacobi (n,a,b,x):
    p = 1
    
    if n == 0:
        return p
    elif n == 1:

        p = 0.5*(a - b + (2+a+b)*x)
        return p
    else:
        p = ((2*n + a + b-1)*((a+b)*(a-b) + x*(2*n + a + b-2)*(2*n + a + b))/(2*(n * (n+a+b) * (2*n + a + b-2)))) * jacobi(n-1, a, b, x) - \
            ((n+a-1)*(n+b-1)*(2*n + a + b)/(n * (n+a+b) * (2*n + a + b-2)))*jacobi(n-2, a, b, x)
        return p

#some hardcoded stuff from matlab
permutations_quads = {}
permutations_quads[1] = np.array([1,2,3,4],dtype = int)-1
permutations_quads[2] = np.array([1,5,2,8,9,6,4,7,3],dtype=int)-1
permutations_quads[3] = np.array([1,5,6,2,12,13,14,7,11,15,16,8,4,10,9,3],dtype=int)
permutations_quads[4] = np.array([1,5,6,7,2,16,17,18,19,8,15,20,21,22,9,
                                  14,23,24,25,10,4,13,12,11,3],dtype=int)-1
permutations_quads[5] = np.array([1,5,6,7,8,2,20,21,22,23,24,9,19,25,26,27,
                                28,10,18,29,30,31,32,11,17,33,34,35,36,12,4
                                 ,16,15,14,13,3],dtype=int)-1
permutations_quads[6] = np.array([1,5,6,7,8,9,2,24,25,26,27,28,29,10,23,30,
                                  31,32,33,34,11,22,35,36,37,38,39,12,21,40,
                                41,42,43,44,13,20,45,46,47,48,49,14,4,19,18,
                                17,16,15,3],dtype = int)-1
