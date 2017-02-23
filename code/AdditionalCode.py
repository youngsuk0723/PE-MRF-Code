# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 14:49:21 2016

@author: youngsuk
"""


# In[]
def genEdgeMatrix(filename_edge, n):
    edge_data = np.genfromtxt(filename_edge, delimiter=',')
    edge_data = edge_data[::2,:]
    
    numberOfEdge, dummy = edge_data.shape
    edge_matrix=np.diag(np.concatenate([np.ones(n)]))
#    edge_matrix=np.zeros([n,n])
    for k in xrange(numberOfEdge):
        i = edge_data[k][0]
        j = edge_data[k][1]
        edge_matrix[j,i]=1
        edge_matrix[i,j]=1
    
    return edge_matrix
# In[805]


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
    AIC = -np.log(alg.det(Theta)) + np.trace(np.dot(Theta, A)) + K
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
    print '\nD = ',D,'T = ', T,'TandD = ', TP,'K = ', K,'P = ', P,'R = ', R,'Score = ', score, 'AIC = ', AIC
            
    return e1, e2, e3, e4 ,e5 , e6  
#
# In[806]
    def drawGraphswithLambda(e1,e2,e3,e4,e5,e6,alpha_set):
    pl.subplot(511)    
#    pl.title('Performance Measures with Perturbed Node Penalty for Local Shift')    
#    pl.title(r'Performance Measures with $\ell_2$ Penalty for Global Shift')
#        pl.title(r'%s, $n_t$ = %s, ($\lambda$, $\beta$) = (%s, %s)'%(Data_type, samplesPerStep, alpha, beta))
    pl.plot(alpha_set, e1)
    pl.subplot(512)
    pl.plot(alpha_set, e2)
    pl.subplot(513)
    pl.plot(alpha_set, e3)
    pl.subplot(514)
    pl.semilogx(alpha_set, e4)
    pl.subplot(515)
#    e5.append(0)
#    e6.append(0)
#    print '\nhere,',e7, e8
    
    e5 = np.insert(e5, 0,1)
    e6 = np.insert(e6, 0,1)
    e5 = np.append(e5, 0)
    e6 = np.append(e6, 0)
    pl.plot(e6,e5,'bo-')
    pl.xlabel('FPR')
    pl.ylabel('TPR')
    
    pl.savefig('error_plots.eps', format = 'eps', bbox_inches = 'tight', dpi = 1000)
    pl.show()
    return 0
    
