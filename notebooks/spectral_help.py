# MIT License

# Copyright (c) 2021 Andy Goldschmidt

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np

def BigOmg(omega, ts):
    '''
    Construct time-domain coordinates for Spectral DMD.
    
    Args:
        omega (`1d-array`): 1d array of frequencies 
        ts (`1d-arary`): 1d array of times
        
    Returns:
        `nd-array`: 2*len(omega) by len(t) array of $[\cos(\vec{omega} t), \sin(\vec{omega} t)]^T$
    '''
    omt = omega.reshape(-1,1)@ts.reshape(1,-1)
    return np.vstack([np.cos(2*np.pi*omt), np.sin(2*np.pi*omt)])


def loss(X, A, omega, ts):
    '''
    Loss function for Spectral DMD.
    
    Returns:
        float: Evaluation of loss function.
    '''
    return np.linalg.norm(X - A@BigOmg(omega,ts))


def grad_loss(X, A, omega, ts):
    '''
    Gradient of the loss function w.r.t. $\omega$.
    '''
    n_omg = len(omega)
    part2 = -4*np.pi*A.T@(X - A@BigOmg(omega,ts))
    grad_res = [0]*n_omg
    for i in range(n_omg):
        grad_res[i] += (-np.sin(2*np.pi*omega[i]*ts)*ts).dot(part2[i,:])
        grad_res[i] += (np.cos(2*np.pi*omega[i]*ts)*ts).dot(part2[n_omg + i,:])
    return np.array(grad_res).reshape(1,-1)


def grad_loss_j(j, X, A, omega, ts):
    '''
    Gradient of the loss function w.r.t. $\omega_j$
    
    Returns:
        float: Gradient of the loss w.r.t $\omega_j$
    '''
    n_omg = len(omega)
    if j > n_omg - 1:
        raise ValueError("Invalid value. Index j={} exceeds len(omega)={}.".format(j,n_omg))
        
    part2 = -4*np.pi*A.T@(X - A@BigOmg(omega,ts))
    grad_res_j = (-np.sin(2*np.pi*omega[j]*ts)*ts).dot(part2[j,:])
    grad_res_j += (np.cos(2*np.pi*omega[j]*ts)*ts).dot(part2[n_omg + j,:])
    return grad_res_j


def residual_j(j, X, A, omega, ts):
    '''
    Residual for the data trajectory X using the model A, $\Omega$. The residual excludes the contribution
    of the frequency $\omega_j$.
    
    TODO:
    * Check the case where A.shape[0] = 1
    
    Returns:
        `ndarray`: residual of shape X.shape[0] by ts.shape[0]
    '''
    n_omega = len(omega)
    if j > n_omega - 1:
        raise ValueError("Invalid value. Index j={} exceeds len(omega)={}.".format(j, n_omega))
    
    j2 = j + n_omega
    indices = np.hstack([np.arange(j), np.arange(j+1,j2), np.arange(j2+1, 2*n_omega)])
    return X - A[:,indices]@(BigOmg(omega,ts)[indices, :])


def update_A(X, omega, ts, threshold, threshold_type):
    # Update A
    # -- This step could get some DMD love
    U,S,Vt = np.linalg.svd(BigOmg(omega, ts), full_matrices=False)
    if threshold_type == 'count':
        r = threshold
    elif threshold_type == 'percent':
        r = np.sum(S/np.max(S) > threshold)
    rU = U[:,:r]
    rS = S[:r]
    rVt = Vt[:r, :]
    return X@np.conj(rVt.T)@np.diag(1/rS)@np.conj(rU.T)


def max_fft_update(result, dt):
        # Real signal means the other half of the hat are complex conjugates
        n = result.shape[1]
        n_sym = n//2 if n % 2 == 0 else n//2 + 1
        
        # Compute fft
        res_hat = np.fft.fft(result, axis=1)
        
        # Get the maximum freq. coordinate considering all data dimensions
        ires = np.argmax(np.sum(np.abs(res_hat[:,:n_sym]), axis=0))
        res_freq = np.fft.fftfreq(n, dt)[:n_sym]
        return res_freq[ires]
        

# Accelerated proximal gradient descent
# -----------------------------------------------------------------------------
def optimizeWithAPGD(x0, func_f, func_g, grad_f, prox_g, beta_f, tol=1e-6, max_iter=1000, verbose=False):
    """
    Optimize with Accelerated Proximal Gradient Descent Method
        min_x f(x) + g(x)
    where f is beta smooth and g is proxiable.
    
    input
    -----
    x0 : array_like
        Starting point for the solver
    func_f : function
        Input x and return the function value of f
    func_g : function
        Input x and return the function value of g
    grad_f : function
        Input x and return the gradient of f
    prox_g : function
        Input x and a constant float number and return the prox solution
    beta_f : float
        beta smoothness constant for f
    tol : float, optional
        Gradient tolerance for terminating the solver.
    max_iter : int, optional
        Maximum number of iteration for terminating the solver.
        
    output
    ------
    x : array_like
        Final solution
    obj_his : array_like
        Objective function value convergence history
    err_his : array_like
        Norm of gradient convergence history
    exit_flag : int
        0, norm of gradient below `tol`
        1, exceed maximum number of iteration
        2, others
    """
    # initial information
    x = x0.copy()
    y = x0.copy()
    g = grad_f(y)
    t = 1.0
    #
    step_size = 1.0/beta_f
    # not recording the initial point since we do not have measure of the optimality
    obj_his = np.zeros(max_iter)
    err_his = np.zeros(max_iter)
    
    # start iteration
    iter_count = 0
    err = tol + 1.0
    while err >= tol:
        #####
        # TODO: complete the accelerate proximal gradient step
        x_new = prox_g(y - step_size*g, step_size)
        t_new = (iter_count - 1)/(iter_count + 2)
        y_new = x_new + t_new*(x_new - x)
        # FISTA version:
        # t_new = (1 + np.sqrt(1+4*t**2))/2
        # y_new = x_new + (t - 1)/t_new*(x_new - x)
        #####
        #
        # update information
        obj = func_f(x_new) + func_g(x_new)
        err = np.linalg.norm(x - x_new)
        #
        np.copyto(x, x_new)
        np.copyto(y, y_new)
        t = t_new
        g = grad_f(y)
        #
        obj_his[iter_count] = obj
        err_his[iter_count] = err
        #
        # check if exceed maximum number of iteration
        iter_count += 1
        if iter_count >= max_iter:
            if verbose:
                print('Proximal gradient descent reach maximum of iteration')
            return x, obj_his[:iter_count], err_his[:iter_count], 1
    #
    return x, obj_his[:iter_count], err_his[:iter_count], 0