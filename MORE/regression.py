import numpy as np

####
def phi(theta): #untransformed theta
    d = theta.shape[0]
    dim_phi =  1 + d + d*(d+1) / 2 # d = dim(theta), dann dim_phi := D = 1 + d + d(d+1)/2
    theta_matrix = theta * np.array([theta]).T
    result_vector = np.array([])
    for i in range(d):
        matrix_row = theta_matrix[i,i:] # take first row beginning from i
        result_vector = np.append(result_vector, matrix_row)
    result_vector = np.append(result_vector, theta) # add theta
    result_vector = np.append(result_vector, [1]) # bias term
    return result_vector

def X(thetas): # compute X matrix for multiple linear regression
    # init X with first theta from thetas (i.e. the list with all thetas)
    # make theta to matrix _m
    phi_theta_0 = phi(thetas[0])
    phi_theta_0_m = np.array([phi_theta_0])
    X = phi_theta_0_m # np.array(theta_m)
    for i in range(1, len(thetas)):
        # make theta to matrix _m
        phi_theta_i = phi(thetas[i])
        phi_theta_i_m = np.array([phi_theta_i])
        X = np.append(X, phi_theta_i_m, 0)
        print("X: " ,X.shape)
    return X

def linear_regression(X, rewards): # X as usual in a linear regression and rewards is the y value
    return np.linalg.lstsq(X, rewards)[0]
####
