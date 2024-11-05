import torch
import numpy as np
from nn import *
from sys_prob import problem_generator
from random import sample
from matplotlib import pyplot as plt

from data_set import data

torch.autograd.set_detect_anomaly(True)

np.random.seed(10000)

def my_train_true_gradient(prob, init_var ,model, num_iteration, num_frame, optimizer):
 
 
    (x_optimizer, lambda_optimizer) = optimizer
    (x_model, lambda_model) = model
    (x, r) = init_var
    
    data_iteration = data()
    L_train_result = []
    Loss_train_result = []
    L_truth_result = []
    obj_truth_result = []
    

    for iter in range(num_iteration):

        for frame in range(num_frame):


            # obtain x loss, lambda_model was not involved in this part
            for param in x_model.parameters():
                param.requires_grad = True
            for param in lambda_model.parameters():
                param.requires_grad = False

            # need more data of lambda to derive a precise x_model

            r_p = lambda_proj(r)
            for _ in range(5):
                _x = x_model(r_p)

                # store one piece of data after some variable is updated
                L_truth = prob(_x.detach().numpy(), r_p.detach().numpy())
                L_truth_result.append(L_truth)
                # record the objective value
                obj_truth = prob.objective(_x.detach().numpy())
                obj_truth_result.append(obj_truth)

                # assume this gradient is correct, need checking
                grad_x = torch.tensor(prob.gradient_x(_x.detach().numpy(), r_p.detach().numpy()),dtype=torch.float32)

                # derive the gradient w.r.t. x
                x_optimizer.zero_grad()
                # L is minimized w.r.t. x
                _x.backward(grad_x)
                x_optimizer.step()
                # here we need to add delta to lambda, however we need to know the range of this delta

                

            for param in x_model.parameters():
                param.requires_grad = False
            r.requires_grad = True
            r_p = lambda_proj(r)
            _x = x_model(r_p)
            partial_grad_x = torch.tensor(prob.gradient_x(_x.detach().numpy(), r_p.detach().numpy()),dtype=torch.float32)
            partial_grad_lambda = torch.tensor(prob.gradient_lambda(_x.detach().numpy()),dtype=torch.float32)
            
            _x.backward(partial_grad_x)

            grad_lambda = r.grad + partial_grad_lambda

            # move to later part, record delta_lambda instead of L_truth
            #data_iteration.append(_x, r_p, L_truth)

            # update lambda_model using lstm
            # equal to learn x_model and Lagrange funcion

            r = r.detach()
            for param in x_model.parameters():
                param.requires_grad = False
            for param in lambda_model.parameters():
                param.requires_grad = True
            
            # derive new r based on current r and current gradient w.r.t. r
            delta_lambda = lambda_model(grad_lambda)
            _r = r + delta_lambda      
            
            r_p = lambda_proj(_r)
            # to derive the gradient w.r.t. lambda /(or projected lambda?) to make sure
            _x = x_model(r_p)
        
            r_p.retain_grad()
            #L = -L_model(_x, r_p)
            
            #L.backward()
            _x.backward(partial_grad_x,retain_graph=True)
            grad_lambda = r_p.grad + torch.tensor(prob.gradient_lambda(_x.detach().numpy()),dtype=torch.float32)
            lambda_optimizer.zero_grad()
            r_p.backward(-grad_lambda)
            lambda_optimizer.step()
            
            #print("L_lambda:",-L.detach().numpy())

            L_truth = prob(_x.detach().numpy(), r_p.detach().numpy())
            data_iteration.append(_x, r_p, L_truth)

            for param in lambda_model.parameters():
                param.requires_grad = False
            delta_lambda = lambda_model(grad_lambda)
            _r = r + delta_lambda

            r_p = lambda_proj(_r)
            _x = x_model(r_p)

            L_truth = prob(_x.detach().numpy(), r_p.detach().numpy())
            data_iteration.append(_x, r_p, L_truth)


            x = _x.detach()
            r = _r.detach()


        

        # here we update L model

    print("lambda:",r_p.detach().numpy())
    print("x:",x.detach().numpy())
    np.save('L_train.npy',np.array(L_train_result))
    np.save('Loss_train.npy',np.array(Loss_train_result))
    np.save('L_truth.npy',np.array(L_truth_result))
    np.save('obj_train.npy',np.array(obj_truth_result))



if __name__ == "__main__":

    
    L = problem_generator()
    result = L.solve()
    out = result.x
    obj = result.fun
   
    class arg_nn:
        hidden_size = 32
        hidden_size_x = 32
    len_x = 5
    len_lambda = 2 * len_x +1
    num_iteration = 500
    num_frame = 60

    x_model = x_LSTM(len_x, len_lambda, arg_nn)
    lambda_model = lambda_LSTM(len_lambda, arg_nn)
    x_optimizer = torch.optim.Adam(x_model.parameters(), lr=0.001)
    lambda_optimizer = torch.optim.Adam(lambda_model.parameters(), lr=0.001)

    r = torch.randn(1,len_lambda)
    x = torch.abs(torch.randn(1,len_x))
    init_var = (x, r)

    model = (x_model, lambda_model)
    optimizer = (x_optimizer, lambda_optimizer)
    
    #my_train(L, init_var, model, num_iteration, num_frame, optimizer)
    my_train_true_gradient(L, init_var, model, num_iteration, num_frame, optimizer)


    print(out)
