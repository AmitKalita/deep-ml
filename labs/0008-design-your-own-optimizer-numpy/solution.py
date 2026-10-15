import numpy as np

def optimizer_step(param, grad, state, lr):
    '''
    Update a parameter using its gradient.
    
    Args:
        param: numpy array - current parameter values (any shape)
        grad: numpy array - gradient of loss w.r.t. param (same shape)
        state: dict - persists between calls, use to store any needed values
                      Example: state = {'step': 5, 'momentum': np.array([...])}
                      First call: state = {} (empty dict)
        lr: float - learning rate
    
    Returns:
        new_param: numpy array - updated parameter (must be same shape as param)
        state: dict - updated state dictionary
    '''
    # TODO: Implement your optimizer update rule
    # Hint: Think about gradient descent and how to use the gradient to update the parameter

    # mu = 0.9
    # v_t = np.zeros_like(param)

    # v_t = mu * v_t + grad 

    # new_param = param - lr * v_t

    # if 'moment' not in state:
    #     state['moment']=v_t

    if "theta" not in state:
        state["theta"] = param
    
    state["theta"] = state["theta"] - lr * grad 
    new_param = state["theta"]

    return new_param, state
