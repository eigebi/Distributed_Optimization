import torch
import numpy as np
from nn_CLSTM import *
from sys_prob import problem_generator
from random import sample
from matplotlib import pyplot as plt

from data_set import data, r_data

torch.autograd.set_detect_anomaly(True)

np.random.seed(10000)
torch.random.manual_seed(10000)



def derive_grad_lambda(prob, _x, r):
    
    alpha = 2
    dr = torch.where(r < -1, -alpha, 0)
    dr = torch.where(r > 1, alpha, dr)
    dr = torch.where((r >= -1) & (r <= 1), alpha*r**(alpha-1), dr)

    partial_grad_lambda_p = torch.tensor(prob.gradient_lambda(_x.detach().numpy()),dtype=torch.float32)

    grad_lambda = dr * partial_grad_lambda_p
  
    return grad_lambda

    



def my_train_true_gradient(prob, init_var ,model, num_iteration, num_frame, optimizer):
 
 
    (x_optimizer, lambda_optimizer) = optimizer
    (x_model, lambda_model) = model
    (x, r) = init_var
    
    data_iteration = data()
    x_data_iteration = r_data()
    L_train_result = []
    Loss_train_result = []
    L_truth_result = []
    obj_truth_result = []
    

    for iter in range(num_iteration):

        for frame in range(num_frame):

            # first update lambda_model          
            
            # batch training of lambda_model
            for param in x_model.parameters():
                param.requires_grad = True
            for param in lambda_model.parameters():
                param.requires_grad = True

            # outer layer is repeating
            # inner update bring N steps forward
            # inner update, from one given initial lambda

            for _ in range(1000):
                reserved_r = r.detach()
                reserved_x = x.detach()
                for _ in range(60):
                    
                    r = r.detach()
                    grad_lambda = derive_grad_lambda(prob, x, r)
                    
                    delta_lambda = lambda_model(grad_lambda)
                    _r = r + delta_lambda
                    r = _r.detach()
                    grad_lambda = derive_grad_lambda(prob, x, r)
                    _r.backward(-grad_lambda, retain_graph=True)

                    r_p = lambda_proj(r)
                    grad_x = torch.tensor(prob.gradient_x(x.detach().numpy(), r_p.detach().numpy()),dtype=torch.float32)
                    delta_x = x_model(grad_x)
                    _x = x + delta_x
                    x = _x.detach()
                    grad_x = torch.tensor(prob.gradient_x(x.detach().numpy(), r_p.detach().numpy()),dtype=torch.float32)
                    _x.backward(grad_x, retain_graph=True)


                    
                lambda_optimizer.step()
                lambda_optimizer.zero_grad()
                x_optimizer.step()
                x_optimizer.zero_grad()
                r_p = lambda_proj(r)

                print("L truth: ", prob(x.detach().numpy(), r_p.detach().numpy()), "obj truth: ", prob.objective(x.detach().numpy()))
                latest_r = r
                latest_x = x
                r = reserved_r
                x = reserved_x
                
            r = latest_r
            x = latest_x

            # update x_model by samlping
            

        # print result each iteration
        print("lambda:",r_p.detach().numpy())
        print("x:",x.detach().numpy())
        res = prob.solve()
        print("opt x: ", res.x)
        print("constraint function: ", prob.check_con(_x.detach().numpy())[-1])
        print("opt obj: ", prob.objective(res.x.reshape(1,-1)))
            


    # end of iterations
    #np.save('L_train.npy',np.array(L_train_result))
    #np.save('Loss_train.npy',np.array(Loss_train_result))
    #np.save('L_truth.npy',np.array(L_truth_result))
    #np.save('obj_train.npy',np.array(obj_truth_result))



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
    num_iteration = 1
    num_frame = 10000

    x_model = x_LSTM(len_x, arg_nn)
    lambda_model = lambda_LSTM(len_lambda, arg_nn)
    x_optimizer = torch.optim.Adam(x_model.parameters(), lr=0.004)
    lambda_optimizer = torch.optim.Adam(lambda_model.parameters(), lr=0.001)

    r = torch.randn(1,len_lambda)
    x = torch.abs(torch.randn(1,len_x))
    init_var = (x, r)

    model = (x_model, lambda_model)
    optimizer = (x_optimizer, lambda_optimizer)
    
    #my_train(L, init_var, model, num_iteration, num_frame, optimizer)
    my_train_true_gradient(L, init_var, model, num_iteration, num_frame, optimizer)


    print(out)
