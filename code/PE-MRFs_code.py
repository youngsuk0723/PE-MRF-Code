
# coding: utf-8

# In[1]:

import numpy as np
from numpy import linalg as la


# In[2]:

def Mmatrix(data, nu, m1, dim1, m2, dim2, m3, dim3, m4, dim4):
    # data is the sufficient statistic data
    num_sam,d_num = np.shape(data)
    # number of sample is 990
    # number of d is 64
    print(num_sam)
    
    #m1: number of Bernoulli sample  = 8
    #dim1: T(x) dimension of Bernoulli sample = 1
    
    #m2: number of Gaussian sample = 8
    #dim2: T(x) dimension of Gaussian sample = 1
        
    #m3: number of Gamma sample = 8
    #dim3: T(x) dimension of Gamma sample = 2
    
    #m4: number of Dirichlet sample  = 8 
     #dim4: T(x) dimension of Dirichlet sample = 3

    endpoint1 = m1 * dim1 # 8
    
    ber_data = data[:,0:endpoint1] # index 0 to 7 are Bernoulli
    #print(ber_data.shape)
    
    endpoint2 = endpoint1 + m2 * 2 * dim2 # 8 + 16 = 24

    # for gaussian case, we only need x part in [x x^2]
    gauss_data = np.zeros(np.shape(ber_data))
    k=0;
    for i in range(endpoint1, endpoint2, 2): # i in range(8,24,2)
        gauss_data[:,k] = data[:,i]
        k=k+1
    
    #print(gauss_data.shape)
    endpoint3 = endpoint2 + m3 * dim3 # 24 + 8*2 = 40

    gamma_data = data[:,endpoint2:endpoint3] # index 24 to 40 are Gamma
    #print(gamma_data.shape)
    endpoint4 = endpoint3 + 2*m4 * dim4 # 40 + 8*3 = 64
    
    # the rest of them are dirichlet distribution
    dirich_data = data[:,endpoint3:endpoint4]
    #print(dirich_data.shape)
    
    vec = np.ones([990,1])
    # construct new data
    new_data = np.concatenate((vec, ber_data, gauss_data, gamma_data, dirich_data), axis=1)
    #print(new_data.shape)
    
    M = np.transpose(np.matrix(new_data))*np.matrix(new_data)/num_sam
    
    return M


# In[3]:

data = np.genfromtxt('sufficient_statistics_0.csv', delimiter=",")
M = Mmatrix(data, 1, 8, 1, 8, 1, 8, 2, 8, 3) 
print(M)


# 
# M correspond to a matrix in the form:
# 
#        [ M11  |   M12 ] 
#        --------------------         
#        [ M21  |   M22 ]
# 

# In[4]:

def Dmatrix(m1, dim1, m2, dim2, m3, dim3, m4, dim4):
    #m1: number of Bernoulli sample  = 8
    #dim1: T(x) dimension of Bernoulli sample = 1
    
    #m2: number of Gaussian sample = 8
    #dim2: T(x) dimension of Gaussian sample = 1
        
    #m3: number of Gamma sample = 8
    #dim3: T(x) dimension of Gamma sample = 2
    
    #m4: number of Dirichlet sample  = 8 
     #dim4: T(x) dimension of Dirichlet sample = 3
    
    # only Bernoulli has value. The rest are continuous random variable
    l0 = 0
    l1 = np.ones(dim1*m1)
    l2 = np.zeros(dim2*m2)
    l3 = np.zeros(dim3*m3)
    l4 = np.zeros(dim4*m4)
    
    #l1 = np.diag(temp);
    temp = np.hstack((l0,l1,l2,l3,l4)) # makes an array in the form of [ l0 l1 l2 l3 l4 ] 
    D = np.diag(temp)
    return D


# In[5]:

data = np.genfromtxt('sufficient_statistics_0.csv', delimiter=",")
M = Mmatrix(data, 1, 8, 1, 8, 1, 8, 2, 8, 3) 
D = Dmatrix(8, 1, 8, 1, 8, 2, 8, 3) 

#np.savetxt('M.csv', M, delimiter=',')


A = M+D
T =np.transpose(A)
error = T - A
print(A)


# In[6]:

############### TEST FOR M matrix and D matrix functions ##################
data = np.genfromtxt('sufficient_statistics_0.csv', delimiter=",")
M = Mmatrix(data, 1, 8, 1, 8, 1, 8, 2, 8, 3) 
# Here, we take dummy variable, nu =1
    # dist_type1: 8 (Bernoulli) 
    # suff_stat for Bernoulli is 1
    
    # dist_type2: 8 (Gaussian) 
    # suff_stat for Gaussian is 1
    
    # dist_type3: 8 (Gamma) 
    # suff_stat for Gamma is 2
    
    # dist_type4: 8 (Dirichlet with k =3) 
    # suff_stat for Dirichlet is k=3 
    #Respectively
    
D = Dmatrix(8, 1, 8, 1, 8, 2, 8, 3) 
# number of Bernoulli sample, Bernoulli T(x) dimension
# number of Gaussian sample, Gaussian T(x) dimension
# number of Gamma sample, Gamma T(x) dimension
# number of Dirichlet sample, Dirichlet T(x) dimension
# respectively

A = M+D
A


# In[7]:

def Block_Extract(Z, row_ind, col_ind, m, dim):
    # m = np.array([1, 8, 8, 8, 8])
    # dim = np.array([1, 1, 1, 2, 3])

    #m[0]: dummy variable, nu =1
    #dim[0]: dummy variable, nu =1
    
    #m[1]: number of Bernoulli sample  = 8
    #dim[1]: T(x) dimension of Bernoulli sample = 1
    
    #m[2]: number of Gaussian sample = 8
    #dim[2]: T(x) dimension of Gaussian sample = 1
        
    #m[3]: number of Gamma sample = 8
    #dim[3]: T(x) dimension of Gamma sample = 2
    
    #m[4]: number of Dirichlet sample  = 8 
    #dim[4]: T(x) dimension of Dirichlet sample = 3
    
    # Obtain New row index
    new_row = 0
    for i in range(len(m)): # iterate through 0 1 2 3 4
        if row_ind < m[i]:
            new_row += dim[i]*row_ind
            Rdim = dim[i]
            break
        row_ind -= m[i]
        new_row += dim[i]*m[i]
    
    # Obtain New column index
    new_col = 0
    for i in range(len(m)):
        if col_ind < m[i]:
            new_col += dim[i]*col_ind
            Cdim = dim[i]
            break
        col_ind -= m[i]
        new_col += dim[i]*m[i]

    return Z[new_row:new_row+Rdim, new_col:new_col+Cdim]
    
#m = np.array([1, 8, 8, 8, 8]) # 1 for dummy variable
#dim = np.array([1, 1, 1, 2, 3]) # 1 for dummy variable, 
#Z = np.ones([57, 57])

#print(Block_Extract(Z, 15, 23, m, dim).shape)


# In[8]:

def Zupdate(theta, U, Eta, m1, dim1, m2, dim2, m3, dim3, m4, dim4):
    # i = 1, ... , 32
    # j = 1, ... , 32
    # Extract number of rows and columns for Z; should be (d+1) and (d+1) respectively
    # for testing, it is a 57 by 57 matrix
    
    #m1: number of Bernoulli sample  = 8
    #dim1: T(x) dimension of Bernoulli sample = 1
    
    #m2: number of Gaussian sample = 8
    #dim2: T(x) dimension of Gaussian sample = 1
        
    #m3: number of Gamma sample = 8
    #dim3: T(x) dimension of Gamma sample = 2
    
    #m4: number of Dirichlet sample  = 8 
     #dim4: T(x) dimension of Dirichlet sample = 3
    
    row,col = theta.shape
    
    dist_num = m1+m2+m3+m4 # 32
    
    m = np.array([1, m1, m2, m3, m4])
    dim = np.array([1, dim1, dim2, dim3, dim4])
    
    temp_matrix = theta+U
    
    B = [ [] for i in range(dist_num+1)] # [  [], [], [], ... [] ]
   
    for i in range(dist_num+1): # iterate from 0 to 32
        for j in range(dist_num+1):
            if i==0 or j==0 or i==j :
                B[i].append(Block_Extract(temp_matrix, i, j, m, dim)) # Helper function called "Block_Extract"
            else:
                gamma = la.norm(Block_Extract(temp_matrix, i, j, m, dim), 'fro')
                #print(gamma)
                if gamma > Eta[i][j]:
                    B[i].append((1 - Eta[i][j]/gamma)*Block_Extract(temp_matrix, i,j, m, dim))
                else:
                    B[i].append(np.zeros(Block_Extract(temp_matrix, i, j, m, dim).shape))
                    
    # now B is [ [B0,0 , B0,1 , ... , B0,31], [B1,0 , B1,1 , ... , B1,31], ..., [B31,0, ..., B31,31]]
    Z_new = np.bmat(B)
    Z_new = (Z_new + np.transpose(Z_new))/2
    
    return Z_new

# [ [B_00] [B_01] ... [B_032] ]


# In[9]:

def ADMM(theta, Z, U, A, K, lamb, W, num_sam, m1, dim1, m2, dim2, m3, dim3, m4, dim4):
    # k stands for number of iterations
    # lamb stands for the lambda, which is lasso parameter 
    # Dimension of theta, U, Z and A is (d+1) x (d+1)
    # for testing, it is 57 by 57 matrix
    
    #m1: number of Bernoulli sample  = 8
    #dim1: T(x) dimension of Bernoulli sample = 1
    
    #m2: number of Gaussian sample = 8
    #dim2: T(x) dimension of Gaussian sample = 1
        
    #m3: number of Gamma sample = 8
    #dim3: T(x) dimension of Gamma sample = 2
    
    #m4: number of Dirichlet sample  = 8 
     #dim4: T(x) dimension of Dirichlet sample = 3
    
    new_theta = theta
    prev_Z = Z
    
    new_U = U
    n = num_sam # 990
    
    mat_dim = theta.shape[0] # 57
    
    rho = 5 # define rho
    eta1 = rho/n # define eta for theta update

    eta2 = (lamb*W)/rho; # define eta2 for Z update
    
    epsilon_abs = 1e-05##### TO DO
    
    epsilon_rel = 1e-04
    
    # Initialize R_k and S_k
    R_k = []
    S_k = []
    epsilon_pri = []
    epsilon_dual = []
    
    for k in range(K):
        temp = np.multiply((Z-U),eta1) - A;
        temp = (temp + np.transpose(temp))/2
#         if k==1:
#             np.savetxt('temp.csv', temp, delimiter=',')
    
        lambmat,Q = la.eig(temp)
        #lambmat is eigenvalue
        #v is eigenvector 
        
        
        # Update theta
        new_theta = (1/(2*eta1))*Q*np.matrix((np.diag(lambmat) + np.sqrt(np.diag(np.power(lambmat,2))+4*eta1*np.identity(len(lambmat)))))*np.transpose(Q)
        
        theta = new_theta # re-assign the theta (correpsond to "theta_k")
        
        # Update Z
        new_Z = Zupdate(new_theta, new_U, eta2, m1, dim1, m2, dim2, m3, dim3, m4, dim4)
    
        #Calculate residual (stopping criteria 1)
            # calculate r^k
        R_k.append(la.norm(theta - new_Z, 'fro'))
        
            # calculate primal epsilon
        epsilon_pri.append( mat_dim * epsilon_abs  + epsilon_rel * max( la.norm(theta, 'fro'), la.norm(new_Z, 'fro') ) )
        
        if R_k[k] <= epsilon_pri[k]:
            sentence = 'Primal Residual True'
        
        # Calculate dual (stopping criteria 2)
            # calculate r^k
        S_k.append(rho * la.norm(new_Z-prev_Z, 'fro'))
        
        prev_Z = new_Z # Book-keeping previous Z (correspond to "Z_k")
        
        #Update U
        new_U = new_U + new_theta - new_Z
        
            # calculate dual epsilon
        epsilon_dual.append( mat_dim * epsilon_abs + epsilon_rel * rho * la.norm(new_U, 'fro'))
        
        if S_k[k] <= epsilon_dual[k]:
            sentence = 'Dual Residual True'
        
        if R_k[k] <= epsilon_pri[k] and S_k[k] <= epsilon_dual[k]:
            sentence = 'primal and dual both met'
            break;
    
    sentence = 'iteration is over'
    print('stopping criteria 1')
    print(R_k[k-1], epsilon_pri[k-1])
    print()
    print('stopping criteria 2')
    print(S_k[k-1], epsilon_dual[k-1])
            
    print(k)
    print(sentence)
    
    return new_theta, new_Z, new_U, R_k, S_k, epsilon_pri, epsilon_dual


# In[10]:

def computeErrors(edge_matrix_actual,edge_matrix_est, Theta, A, e1,e2,e3,e4,e5,e6):
    n = edge_matrix_actual.shape[0]
    D = np.where(edge_matrix_est != 0)[0].shape[0]#len(numpy.where(S_est == 0)[0])
    T = np.where(edge_matrix_actual != 0)[0].shape[0]
#            print np.where(S_actual != 0)[0]
    TP = float(np.where(np.logical_and(edge_matrix_actual,edge_matrix_est) == True)[0].shape[0])
    
#    T = np.count_nonzero(edge_matrix_actual)
#    D = np.count_nonzero(edge_matrix_est)
#    TandD = float(np.count_nonzero( (edge_matrix_actual != 0) & (edge_matrix_est != 0) ) )
    P = TP/D
    R = TP/T
    score = 2* P*R/(P+R)
    offDiagDiff = edge_matrix_actual - edge_matrix_est
    offDiagDiff = offDiagDiff - np.diag(np.diag(offDiagDiff))
#    S_diff = (S_est - S_previous)  
#    S_diff = S_diff - np.diag(np.diag(S_diff))
#    ind = (S_diff < 1e-2) & (S_diff > - 1e-2)
#    S_diff[ind] = 0    
#    K = np.count_nonzero(S_diff)
    K = D
    AIC = -np.log(la.det(Theta)) + np.trace(np.dot(Theta, A)) + K
    TPR = TP/T
    FPR = (D-TP)/(np.square(n)-T)
    e1.append(P)
    e2.append(R)        
    e3.append(score)
#    K = float(np.where(np.logical_and((S_est>0) != (S_previous>0), S_est>0) == True)[0].shape[0])
    e4.append(AIC) #AIC
    e5.append(TPR) #AIC
    e6.append(FPR) #AIC
    
#    e4.append(alg.norm(S_est -  S_previous, 'fro'))
    print ('\nD = ',D,'T = ', T,'TandD = ', TP,'K = ', K,'P = ', P,'R = ', R,'Score = ', score, 'AIC = ', AIC)
            
    return e1, e2, e3, e4 ,e5 , e6  


# In[11]:

# In[]
def genEdgeMatrix(filename_edge, n):
    edge_data = np.genfromtxt(filename_edge, delimiter=',')
    edge_data = edge_data[::2,:]
    
    numberOfEdge, dummy = edge_data.shape

    edge_matrix=np.diag(np.concatenate([np.ones(n)]))
#    edge_matrix=np.zeros([n,n])
    for k in range(numberOfEdge):
        i = edge_data[k][0]
        j = edge_data[k][1]
        edge_matrix[j,i]=1
        edge_matrix[i,j]=1
    
    return edge_matrix
# In[805]


# In[12]:

def createEmatrix(theta, m1, dim1, m2, dim2, m3, dim3, m4, dim4):
    #m1: number of Bernoulli sample  = 8
    #dim1: T(x) dimension of Bernoulli sample = 1
    
    #m2: number of Gaussian sample = 8
    #dim2: T(x) dimension of Gaussian sample = 1
        
    #m3: number of Gamma sample = 8
    #dim3: T(x) dimension of Gamma sample = 2
    
    #m4: number of Dirichlet sample  = 8 
     #dim4: T(x) dimension of Dirichlet sample = 3
    
    E = np.zeros([33,33])
    dist_num = 32;
    B = [ [] for i in range(dist_num+1)] # [  [], [], [], ... [] ]
    m = np.array([1, m1, m2, m3, m4])
    dim = np.array([1, dim1, dim2, dim3, dim4])
    temp_vec = []
    for i in range(dist_num+1): # iterate from 0 to 32
        for j in range(dist_num+1):
            B[i].append(Block_Extract(theta, i, j, m, dim)) # Helper function called "Block_Extract"
            temp = la.norm(Block_Extract(theta, i, j, m, dim), 'fro')
            temp_vec.append(temp)
            if np.absolute(temp) >= 1e-2:
                E[i,j] = 1
            else:
                E[i,j] = 0

    return E[1:33,1:33]
            


# In[101]:

# np.set_printoptions(suppress=True, precision = 3)

###################################TEST CODE######################################
k = [i for i in range(0,5)]
for i in range(3,4):
    file_name = 'sufficient_statistics_'+str(k[i])+'.csv'
    file_name2 = 'modifiedEdge_'+str(k[i])+'.csv'
    
    print('test for')
    print(file_name)
    print()
    print(file_name2)
    print()
    
    data = np.genfromtxt(file_name, delimiter=",")
    M = Mmatrix(data, 1, 8, 1, 8, 1, 8, 2, 8, 3)
    D = Dmatrix(8, 1, 8, 1, 8, 2, 8, 3) 
    # number of Bernoulli sample, Bernoulli T(x) dimension
    # number of Gamma sample, Gamma T(x) dimension
    # number of Gaussian sample, Gaussian T(x) dimension
    # number of Dirichlet sample, Dirichlet T(x) dimension
    # respectively
    A = M+D
    A = np.matrix(A) + np.identity(57)
    
    #A = np.random.randn(57,57)
    
    A = (A + np.transpose(A))/2
    
    Lamvec, Vec = la.eig(A)
    min_lamb = min(Lamvec)
    #A = A + np.identity(57)*(-1*(min_lamb) + 1)
    
    A = (A + np.transpose(A))/2
    
    Lamvec,Vec = la.eig(A)
    min_lamb = min(Lamvec)
    if min_lamb> 0:
        print('PSD!!!')
        print(min_lamb)
    else:
        print('Negatve Definite')
        print(min_lamb)
    T =np.transpose(A)
    error = T - A
    print('error')
    print(la.norm(error, 'fro'))

    #np.savetxt('A.csv', A, delimiter=',')

    ### initialize Theta ###
    theta = np.ones([57, 57])*0.1  # theta is a 57 x 57 matrix with each entries with 0.1

    #print(theta)
    
    Z = np.ones([57,57])*0.2 #57x57 identity matrix ------- when I made this matrix to 0 matrix, in theta update, it calculates (Z - U) as (0 mat - 0 mat) gives error
    U = np.ones([57, 57])*0.1 #0 matirx
    K = 600 #number of iteration
    
    Mvec = np.zeros([33,1])
    for i in range(0,33):
        if i < 17:
            Mvec[i] = 1
        elif i < 25:
            Mvec[i] = 2
        else:
            Mvec[i] = 3

    #print(Mvec)
    #print(np.transpose(Mvec))
    Wmatrix = np.sqrt(Mvec * np.transpose(Mvec))
    print('Wmatrix is ')
    print(Wmatrix)
    W = Wmatrix 
    
    #np.savetxt('Wmatrix.csv', W, delimiter=',')

    number_of_sample,d_num = np.shape(data)
    set_length = 5
    lamb = np.logspace(-2, 2, set_length) #where set_length is the number of points in the ROC graph
    #lamb = np.array([0,0]);
    
    TPRvec = [[] for p in range(len(lamb))]
    FPRvec = [[] for p in range(len(lamb))]
    
    for p in range(len(lamb)):
        print('lambda value is')
        print(lamb[p])
        optimal_theta, optimal_Z, optimal_U, R_k, S_k, epsilon_pri, epsilon_dual = ADMM(theta, Z, U, A, K, lamb[p], W, number_of_sample, 8, 1, 8, 1, 8, 2, 8, 3)
        '''
        print('Number of iteration was ' + str(num_iter))
        print()
        print('Optimal Theta is')
        print(optimal_theta)
        print(optimal_theta.shape)
        print()
        print('Optimal Z is')
        print(optimal_Z)
        print()
        print('Optimal U is')
        print(optimal_U)
        print()

        print('Residual (Primal) is')
        print(R_k)
        print()
        print('Residual (Dual) is')
        print(S_k)
        print()
        print('epsilon_pri is')
        print(epsilon_pri)
        print()
        print('epsilon_dual is')
        print(epsilon_dual)
        '''
        print()
        print('Optimal Theta is')
        print(optimal_theta)
        print()
        #print('optimal Z is')
        #print(optimal_Z)
        
#         print()
#         print('optimal U is')
#         print(optimal_U)
#         print()
        
        Ematrix = createEmatrix(optimal_theta, 8, 1, 8, 1, 8, 2, 8, 3)
      
        trueEmatrix = genEdgeMatrix(file_name2, 32)
    
        #print(trueEmatrix.shape)
        #print(Ematrix.shape)
        e1 = []
        e2 = []
        e3 = []
        e4 = []
        e5 = []
        e6 = []

        error1, error2, error3, error4, error5, error6 = computeErrors(trueEmatrix, Ematrix, optimal_theta, A, e1,e2,e3,e4,e5,e6)
        
        TPRvec[p].append(error5)
        FPRvec[p].append(error6) 
        #print(error5)
        #print()
        #print(error6)
    print(TPRvec[0])
    print(FPRvec[0])


# In[98]:




#np.savetxt('optimal_theta.csv', optimal_theta, delimiter=',')


print(A)
print()
print('A Inverse is')
print(la.inv(A))
#print(np.diag(la.inv(A)))
print()
print('Optimal Z is')
print(optimal_Z)
print('Optimal theta is')
print(optimal_theta)
print('optimal U is')
print(optimal_U)

##

print('A is PSD??')
L,V = la.eig(A)
if min(L) > 0:
    print('A is PSD')
    print(min(L))


# In[260]:

TPRvec = [[] for i in range(30)]
for i in range(30):
    TPRvec[i].append(5)
    TPRvec[i].append(10)
    
print(TPRvec)


# In[232]:

#################################Code for Test Z update#####################################

######################### Initialize values ############################
m1 = 8
dim1 = 1
m2 = 8
dim2 = 1
m3 = 8
dim3 = 2
m4 = 8
dim4 = 3

m = np.array([1, m1, m2, m3, m4])
dim = np.array([1, dim1, dim2, dim3, dim4])

### initialize Theta ###
theta = np.ones([57, 57])*0.1  # theta is a 57 x 57 matrix with each entries with 0.1

endpoint1 = m1 * dim1 +1 # 9
endpoint2 = endpoint1 + m2 * dim2 # 9 + 8 = 17
endpoint3 = endpoint2 + m3 * dim3 # 17 + 16 = 33
endpoint4 = endpoint3 + m4 * dim4 # 33 + 24 = 57

endpoint = np.array([1, endpoint1, endpoint2, endpoint3, endpoint4]) # 1, 9, 17, 33, 57

## commented out the previously tested part
#for i in range(4):
#    theta[endpoint[i]:,endpoint[i]:] = theta[endpoint[i]:,endpoint[i]:] + 1

np.savetxt('theta.csv', theta, delimiter=',')   # X is an array

#theta2 = np.zeros([57,57])
#for i in range(4):
#    theta2[endpoint[i]:endpoint[i+1],endpoint[i]:endpoint[i+1]] = theta2[endpoint[i]:endpoint[i+1],endpoint[i]:endpoint[i+1]] + 1

#np.savetxt('theta2.csv', theta2, delimiter=',')   # X is an array

'''
### initialize W ###
W = np.ones([33, 33])
W[1:9,9:17] = 1/2
W[1:9,17:25] = 3/4
W[1:9,25:33] = 7/8

W[9:17,1:9] = 1/2
W[9:17,17:25] = 7/8
W[9:17,25:33] = 3/4

W[17:25,1:9] = 3/4
W[17:25,9:17] = 7/8
W[17:25,25:33] = 1/2

W[25:33,1:9] = 7/8
W[25:33,9:17] = 3/4
W[25:33,17:25] = 1/2
'''
Mvec = np.zeros([33,1])
for i in range(0,33):
    if i < 17:
        Mvec[i] = 1
    elif i < 25:
        Mvec[i] = 2
    else:
        Mvec[i] = 3

#print(Mvec)
#print(np.transpose(Mvec))
Wmatrix = np.sqrt(Mvec * np.transpose(Mvec))
print('Wmatrix is ')
print(Wmatrix)

np.savetxt('Wmatrix.csv', W, delimiter=',')  

U = np.ones([57, 57])*0.1 #0 matirx
K = 30 #number of iteration
lamb = 1
n = 990

rho = 1 # define rho
eta1 = rho/n # define eta for theta update
eta2 = (lamb*W)/rho;

#################################Test Z update#####################################
print(theta)
# Update Z
for k in range(3):
    print(k)
    new_Z = Zupdate(theta, U, eta2, m1, dim1, m2, dim2, m3, dim3, m4, dim4)

np.savetxt('Z3.csv', new_Z, delimiter=',')   # X is an array


# In[ ]:



