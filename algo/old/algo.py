import torch
import numpy as np
from nn import *
from sys_prob import problem_generator
from random import sample
from matplotlib import pyplot as plt

from data_set import data, r_data

torch.autograd.set_detect_anomaly(True)

np.random.seed(10000)
torch.random.manual_seed(10000)



def derive_grad_lambda(prob, x_model, r):
    for param in x_model.parameters():
        param.requires_grad = False
    r.requires_grad = True
    r_p = lambda_proj(r)
    _x = x_model(r_p)
    partial_grad_x = torch.tensor(prob.gradient_x(_x.detach().numpy(), r_p.detach().numpy()),dtype=torch.float32)
    partial_grad_lambda_p = torch.tensor(prob.gradient_lambda(_x.detach().numpy()),dtype=torch.float32)
    r_p.backward(partial_grad_lambda_p, retain_graph=True)
    _x.backward(partial_grad_x)
    grad_lambda = r.grad
  
    return grad_lambda

    



def my_train_true_gradient(prob, init_var ,model, num_iteration, num_frame, optimizer):
 
 
    (x_optimizer, lambda_optimizer) = optimizer
    (x_model, lambda_model) = model
    (x, r) = init_var
    

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
                param.requires_grad = False
            for param in lambda_model.parameters():
                param.requires_grad = True

            # outer layer is repeating
            # inner update bring N steps forward
            # inner update, from one given initial lambda

            for _ in range(10):
                reserved_r = r.detach()
                for _ in range(5):
                    r = r.detach()
                    
                    grad_lambda = derive_grad_lambda(prob, x_model, r)
                    r = r.detach()
                    delta_lambda = lambda_model(grad_lambda)
                    _r = r + delta_lambda
                    r = _r.detach()
                    grad_lambda = derive_grad_lambda(prob, x_model, r)
                    _r.backward(-grad_lambda, retain_graph=True)
                    x_data_iteration.append(lambda_proj(r))

                    '''
                    r_p = lambda_proj(r)
                    _x = x_model(r_p)
                    grad_x = torch.tensor(prob.gradient_x(_x.detach().numpy(), r_p.detach().numpy()),dtype=torch.float32)
                    _x.backward(grad_x, retain_graph=True)
                    '''

                    
                lambda_optimizer.step()
                lambda_optimizer.zero_grad()
                #x_optimizer.step()
                #x_optimizer.zero_grad()
                r_p = lambda_proj(r)
                _x = x_model(r_p)
                print("L truth: ", prob(_x.detach().numpy(), r_p.detach().numpy()), "obj truth: ", prob.objective(_x.detach().numpy()))
                latest_r = r
                r = reserved_r
            r = latest_r

            # print result each iteration
            print("lambda:",r_p.detach().numpy())
            print("x:",_x.detach().numpy())
            res = prob.solve()
            print("constraint function: ", prob.check_con(_x.detach().numpy()))
            print("opt obj: ", prob.objective(res.x.reshape(1,-1)))




            # update x_model by samlping
            for param in x_model.parameters():
                param.requires_grad = True
            for param in lambda_model.parameters():
                param.requires_grad = False
            for _ in range(50):
                sampled_id = sample(range(len(x_data_iteration.r_p)),min(10,len(x_data_iteration.r_p)))
                r_p_data = torch.tensor(x_data_iteration.r_p,dtype=torch.float32)[sampled_id]
                _x = x_model(r_p_data)
                grad_x = torch.tensor(prob.gradient_x(_x.detach().numpy(), r_p.detach().numpy()), dtype=torch.float32)
                x_optimizer.zero_grad()
                _x.backward(grad_x)
                x_optimizer.step()
            r_p = lambda_proj(r)
            _x = x_model(r_p)
            print("L truth: ", prob(_x.detach().numpy(), r_p.detach().numpy()), "obj truth: ", prob.objective(_x.detach().numpy()))
                

       


    # end of iterations
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
    num_frame = 10

    x_model = x_LSTM(len_x, len_lambda, arg_nn)
    lambda_model = lambda_LSTM(len_lambda, arg_nn)
    x_optimizer = torch.optim.Adam(x_model.parameters(), lr=0.001)
    lambda_optimizer = torch.optim.Adam(lambda_model.parameters(), lr=0.002)

    r = torch.randn(1,len_lambda)
    x = torch.abs(torch.randn(1,len_x))
    init_var = (x, r)

    model = (x_model, lambda_model)
    optimizer = (x_optimizer, lambda_optimizer)
    
    #my_train(L, init_var, model, num_iteration, num_frame, optimizer)
    my_train_true_gradient(L, init_var, model, num_iteration, num_frame, optimizer)


    print(out)
