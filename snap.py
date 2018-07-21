import numpy as np
import numpy.matlib
def snap(w, S):
    # 
    # This functions finds the polynomials p_i(t) of degree 7 that pass through n waypoints. 
    #
    waypoints = np.matrix.transpose(w)
    end = S[0,:].size
    T = S[0,1:end] - S[0,0:end-1]
    n = waypoints[:,0].size - 1
    m = waypoints[0,:].size
    if T[0,:].size != n:
        print('incorrect data')
    A = np.zeros((7*n,7*n))
    B = np.zeros((7*n,m))
    #print(np.shape(A), np.shape(B))
    # Calculating Matrices A and B
    # Rows 1 to n-1
    for i in range(n-1):
        A[i, :] = np.concatenate((A[i, 0:7*(i+1)-7], [1/T[0,i], 2/T[0,i], 3/T[0,i], 4/T[0,i], 5/T[0,i],\
                                                  6/T[0,i], 7/T[0,i], -1/T[0,i+1]], A[i, 7*(i+1)+1:7*n]))
    # Rows n to 2n-2
    for i in range (n-1):
        A[n-1+i, :] = np.concatenate((A[i, 0:7*(i+1)-6], [2/T[0,i]**2, 6/T[0,i]**2, 12/T[0,i]**2, 20/T[0,i]**2,\
                                      30/T[0,i]**2, 42/T[0,i]**2, 0, -2/T[0, i+1]**2], A[i, 7*(i+1)+2:7*n]))
        
    # Rows 2n-1 to 3n-3
    for i in range (n-1):
        A[2*n-2+i, :] = np.concatenate((A[i, 0:7*(i+1)-5], [6/T[0,i]**3, 24/T[0,i]**3, 60/T[0,i]**3, 120/T[0,i]**3,\
                                                210/T[0,i]**3, 0, 0, -6/T[0,i+1]**3], A[i, 7*(i+1)+3:7*n]))
    # Rows 3n-2 to 4n-4
    for i in range (n-1):
        A[3*n-3+i, :] = np.concatenate((A[i, 0:7*(i+1)-4], [24/T[0,i]**4, 120/T[0,i]**4, 360/T[0,i]**4,\
                       840/T[0,i]**4, 0, 0, 0, -24/T[0,i+1]**4], A[i, 7*(i+1)+4:7*n]));
    
    # Rows 4n-3 to 5n-5
    for i in range (n-1):
        A[4*n-4+i, :] = np.concatenate((A[i, 0:7*(i+1)-3], [120/T[0,i]**5, 720/T[0,i]**5, 2420/T[0,i]**5,\
                       0, 0, 0, 0, -120/T[0,i+1]**5], A[i, 7*(i+1)+5:7*n]))    
        
    # Rows 5n-4 to 6n-6
    for i in range (n-1):
        A[5*n-5+i, :] = np.concatenate((A[i,0:7*(i+1)-2], [720/T[0,i]**6, 5040/T[0,i]**6, 0, 0, 0, 0, 0,\
                                -720/T[0,i+1]**6], A[i, 7*(i+1)+6:7*n]))
    #
    # Matrix B is already zero from 1 to 6n-6
    #    
    for i in range (n):
        A[6*n-6+i, :] = np.concatenate((A[6*n-6+i,0:7*(i+1)-7], [1, 1, 1, 1, 1, 1, 1], A[6*n-6+1, 7*(i+1):7*n]))
        #print(waypoints)
        B[6*n-6+i, :] = waypoints[i+1,:] - waypoints[i,:] 
        
    #
    # More A
    #
    A[7*n-6,0] = 1;
    A[7*n-5,1] = 1;
    A[7*n-4,2] = 1;
    A[7*n-3,:] = np.concatenate((A[7*n-3,0:7*(n)-7], [1, 2, 3, 4, 5, 6, 7]))
    A[7*n-2,:] = np.concatenate((A[7*n-2,0:7*(n)-6], [2, 6, 12, 20, 30, 42]))
    A[7*n-1,:] = np.concatenate((A[7*n-1,0:7*(n)-5], [6, 24, 60, 120, 210]))
    #
    XX = np.linalg.solve(A, B)
    #
    X = np.zeros((8*n,m))
    for i in range (n):
        X[8*i,:] = waypoints[i,:]
        X[8*i+1:8*(i+1),:] = XX[7*i:7*i+7,:]
    #
    #
    # Coefficients for the derivatives of the polynomials
    #
    c = np.matrix([[1., 2., 3., 4., 5., 6., 7.]])
    nn = XX[:,0].size
    b = np.matrix([[]])
    for i in range (int(nn/7)):
        bb = c/T[0,i]
        b = np.concatenate((b, bb), axis=1)
    X_b = np.multiply(XX, np.matlib.repmat(b.T,1,3))
    #print(np.matlib.repmat(b.T,1,3).size)
    #
    # Coefficients for the second derivatives of the polynomials
    #
    XXX = XX
    for i in range (int(nn/7)):
        XXX = np.delete(XXX, 6*i, 0)    
    #print(XXX)    
    cc = np.matrix([[2., 6.,  12.,  20.,  30.,  42.]])    
    bprime = np.matrix([[]])
    for i in range (int(nn/7)):
        bb = cc/T[0,i]**2
        bprime = np.concatenate((bprime, bb), axis=1)
    X_bb = np.multiply(XXX, np.matlib.repmat(bprime.T,1,3))  
    return (X, X_b, X_bb)        
    
