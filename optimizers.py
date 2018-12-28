import autograd.numpy as np

def momentum(g, velocity, step_size=0.1, mass=0.9):
    velocity = mass * velocity - (1.0 - mass) * g
    update = -step_size * velocity
    return([update , velocity])

#rmsdrop
def rmsdrop( g, avg_sq_grad, step_size=0.1, gamma=0.9, eps=10**-8):
    avg_sq_grad = avg_sq_grad * gamma + g**2 * (1 - gamma)
    update = -step_size * g/(np.sqrt(avg_sq_grad) + eps)
    return([update ,avg_sq_grad])

#adam
def adam(g,m,v,t,step_size=0.005, b1=0.9, b2=0.999, eps=10**-8):
    m = (1 - b1) * g      + b1 * m  # First  moment estimate.
    v = (1 - b2) * (g**2) + b2 * v  # Second moment estimate.
    mhat = m / (1 - b1**(t + 1))    # Bias correction.
    vhat = v / (1 - b2**(t + 1))
    update = -step_size*mhat/(np.sqrt(vhat) + eps)
    return([update ,m,v])