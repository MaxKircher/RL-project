import numpy as np

'''
    Recap:
     - MORE learns a objective function by defining a quadratic surrogate
       and locally approximate the objective function with that surrugate
     - The quadratic surrugate is learned by linear regression

    Phi is a feature function that
     - (takes a vector) theta:
        - parameter vector of dimension d := deg(polynom) * state_dim + 1
        - is in list format, i.e. untransformed unlike the thetas in policy.py for the dot product
     - Returns:
        1. A bias term r_0 (dimension = 1)
        2. A linear term r (dimension = d)
        3. A quadratic term R (symmetric (state_dim x state_dim)-Matrix) (dimension = d*(d+1)/2)
     - Has dimension  1 + d + d*(d+1) / 2

    Convention on result_vector that contains 1. - 3.
     - Entries:
        1               to d*(d+1)/2        ->
        d*(d+1)/2 + 1   to d*(d+1)/2 + d    ->
        d*(d+1)/2 + d+1 to d*(d+1)/2 + d+2  ->
'''
def __phi__(theta):
    # dimension of theta
    d = theta.shape[0]

    # dimension of phi
    dim_phi =  1 + d + d*(d+1) / 2

    # construct (d x d)-Matrix
    theta_matrix = theta * np.array([theta]).T
    result_vector = np.array([])
    for i in range(d):
        matrix_row = theta_matrix[i,i:] # take first row beginning from i
        result_vector = np.append(result_vector, matrix_row)
    result_vector = np.append(result_vector, theta) # add theta
    result_vector = np.append(result_vector, [1]) # bias term
    return result_vector

'''
    Linear regression is of format: rewards = X*beta

    X computes the X matrix for multpile linear regression
    init X with first theta from thetas (i.e. the list with all thetas)
    make theta to matrix _m
'''
def X(thetas):
    # init X with first theta from thetas (i.e. the list with all thetas)
    # make theta to matrix _m
    phi_theta_0 = __phi__(thetas[0])
    phi_theta_0_m = np.array([phi_theta_0])
    X = phi_theta_0_m # np.array(theta_m)
    for i in range(1, len(thetas)):
        # make theta to matrix _m
        phi_theta_i = __phi__(thetas[i])
        phi_theta_i_m = np.array([phi_theta_i])
        X = np.append(X, phi_theta_i_m, 0)
        print("X: " ,X.shape)
    return X

def linear_regression(X, rewards): # X as usual in a linear regression and rewards is the y value
    return np.linalg.lstsq(X, rewards)[0]

def R_squared():
    return -1
