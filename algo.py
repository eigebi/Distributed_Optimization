import torch
import numpy as np
from nn import *
from sys_prob import problem_generator
from random import sample
from data_set import data

torch.autograd.set_detect_anomaly(True)

np.random.seed(10000)

def my_train(prob, init_var ,model, num_iteration, num_frame, optimizer):
 
 
    (x_optimizer, L_optimizer, lambda_optimizer) = optimizer
    (x_model, L_model, lambda_model) = model
    (x, r) = init_var
    
    


    loss_MSE = torch.nn.MSELoss()
    data_iteration = data()

    r.requires_grad = True

    # freeze x model and learn L model
    for iteration in range(num_iteration):
        # we need the latest gradient w.r.t. r within x and lambda update
        # in frames, L_model is fixed
        for param in L_model.parameters():
            param.requires_grad = False
        
        L_x = 0
        L_lambda = 0
        x_model.zero_grad()
        lambda_model.zero_grad()
        for frame in range(num_frame):
            #for param in x_model.parameters():
            #    param.requires_grad = False

           
            r_p = lambda_proj(r)
            L = L_model(x, r_p)
            L.backward()
            grad_lambda_x = r.grad

            delta_x = x_model(grad_lambda_x)
            _x = x + delta_x

            r.grad.zero_()
            r_p = lambda_proj(r)
            L = L_model(_x, r_p)
            # keep graph for x and lambda update
            L.backward(retain_graph=True)
            grad_lambda_lambda = r.grad
            r.grad.zero_()
            delta_lambda = lambda_model(grad_lambda_lambda)
            _r = r + delta_lambda

            r_p = lambda_proj(_r)
            L_x += L_model(_x, r_p)
            L_lambda -= L_model(_x, r_p)
            L_truth = prob(_x.detach().numpy(), r_p.detach().numpy())
            data_iteration.append(_x, r_p, L_truth)
            r = _r.clone()
            x = _x.clone()

        L_lambda.backward(retain_graph=True)
        L_x.backward()
        lambda_optimizer.step()
        x_optimizer.step()

        for _ in range(30):
            L_optimizer.zero_grad()

            id_sample = sample(range(len(data_iteration.x_past)),min(50,len(data_iteration.x_past)))
            
            x_data = torch.tensor(data_iteration.x_past,dtype=torch.float32)[id_sample]
            r_data = torch.tensor(data_iteration.lambda_past,dtype=torch.float32)[id_sample]
            L_truth = torch.tensor(data_iteration.L_past,dtype=torch.float32)[id_sample]

            loss_L = loss_MSE(L_model(x_data.view(-1,len_x),r_data.view(-1,len_lambda)),L_truth.view(-1,1))
            loss_L.backward()
            L_optimizer.step()
        

        print("L loss", loss_L.detach().numpy(),'delta',out-x[0].detach().numpy())

        # here we update L model

    print("lambda:",r_p.detach().numpy())
    print("x:",x.detach().numpy())



if __name__ == "__main__":

    
    L = problem_generator()
    out = L.solve().x
   
    class arg_nn:
        hidden_size = 32
        hidden_size_x = 16
    len_x = 5
    len_lambda = 2 * len_x +1
    num_iteration = 100
    num_frame = 20

    x_model = x_LSTM(len_x, len_lambda, arg_nn)
    L_model = L_MLP(len_x, len_lambda, arg_nn)
    lambda_model = lambda_LSTM(len_lambda, arg_nn)
    x_optimizer = torch.optim.SGD(x_model.parameters(), lr=0.001)
    L_optimizer = torch.optim.SGD(L_model.parameters(), lr=0.001)
    lambda_optimizer = torch.optim.SGD(lambda_model.parameters(), lr=0.001)

    r = torch.randn(1,len_lambda)
    x = torch.randn(1,len_x)
    init_var = (x, r)

    model = (x_model, L_model, lambda_model)
    optimizer = (x_optimizer, L_optimizer, lambda_optimizer)
    
    my_train(L, init_var, model, num_iteration, num_frame, optimizer)

    print(out)
