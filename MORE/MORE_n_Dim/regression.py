from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np

def __phi__(thetas):
    poly = PolynomialFeatures(2)
    return poly.fit_transform(thetas)

def linear_regression(thetas, rewards):
    features = __phi__(thetas)
    #print("features: ", features)
    reg = LinearRegression().fit(features, rewards)
    return reg.coef_


def compute_quadratic_surrogate(thetas, rewards, d):

    beta_hat = linear_regression(thetas, rewards)
    print("beta_hat: ", beta_hat)
    r = beta_hat[1 : d+1]
    #r0 = beta_hat[0]

    R_param = beta_hat[d+1:]
    # Construct R matrix

    R = np.eye(d) # siehe Kommentar für polynomial policy von Grad 2

    j = 0
    for i in range(d):
        R[i, i:] = R_param[j:j+d-i]
        R[i:, i] = R_param[j:j+d-i]
        j += d-i


    # print(R - R.T) # if it's not a Null-Matrix, something went wrong

    return R, r
