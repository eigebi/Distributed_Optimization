import torch
import numpy as np
from nn import *
from nn_no_grad import L_LSTM
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

    

    # freeze x model and learn L model
    for iteration in range(num_iteration):
        # we need the latest gradient w.r.t. r within x and lambda update
        # in frames, L_model is fixed
        for param in L_model.parameters():
            param.requires_grad = False
        
        for frame in range(num_frame):
            # obtain x loss, lambda_model was not involved in this part
            for param in x_model.parameters():
                param.requires_grad = True
            for param in lambda_model.parameters():
                param.requires_grad = False
            # here r should be detached
            r_p = lambda_proj(r)
            for _ in range(20):
                _x = x + x_model(r_p)
                L = L_model(_x, r_p)
                # store one piece of data after some variable is updated
                L_truth = prob(_x.detach().numpy(), r_p.detach().numpy())
                data_iteration.append(_x, r_p, L_truth)

                #L_x = L_x + L
                # derive the gradient w.r.t. r
                x_optimizer.zero_grad()
                L.backward()
                x_optimizer.step()
                x = _x.detach()
                #print("L_x:",L.detach().numpy())

            for param in x_model.parameters():
                param.requires_grad = False
            r.requires_grad = True
            r_p = lambda_proj(r)
            _x = x + x_model(r_p)
            L = L_model(_x, r_p)
            
            L.backward()
            grad_lambda = r.grad

            L_truth = prob(_x.detach().numpy(), r_p.detach().numpy())
            data_iteration.append(_x, r_p, L_truth)


            # update r
            # current r
            r = r.detach()
            x = _x.detach()
            for param in x_model.parameters():
                param.requires_grad = False
            for param in lambda_model.parameters():
                param.requires_grad = True
            
            # derive new r based on current r and current gradient w.r.t. r
            delta_lambda = lambda_model(grad_lambda)
            _r = r + delta_lambda
            #_r.retain_grad()
            
            lambda_optimizer.zero_grad()
            r_p = lambda_proj(_r)
            _x = x + x_model(r_p)
            L = -L_model(_x, r_p)
            
            L.backward()
            lambda_optimizer.step()
            #print("L_lambda:",-L.detach().numpy())

            L_truth = prob(_x.detach().numpy(), r_p.detach().numpy())
            data_iteration.append(_x, r_p, L_truth)

            for param in lambda_model.parameters():
                param.requires_grad = False
            delta_lambda = lambda_model(grad_lambda)
            _r = r + delta_lambda

            r_p = lambda_proj(_r)
            _x = x + x_model(r_p)

            L_truth = prob(_x.detach().numpy(), r_p.detach().numpy())
            data_iteration.append(_x, r_p, L_truth)


            x = _x.detach()
            r = _r.detach()

            

            '''
            # mind the minus sign
            grad_lambda = -_r.grad
            
            # prepare x and r for the next frame
            r.requires_grad = True
            delta_lambda.retain_grad()
            _r = r + delta_lambda         
            r_p = lambda_proj(_r)
            delta_x = x_model(r_p)
            _x = x + delta_x

            
            

            L = L_model(_x, r_p)
            L_x = L_x + L
            # here the loss is for x only
            L.backward(retain_graph=True)
            

            #L_lambda = L_lambda - L

            grad_lambda = r.grad + delta_lambda.grad
            r = _r.detach()
            delta_lambda = lambda_model(grad_lambda)
            # cut the gradient flow, make x and r pure data
            x = _x.detach()


            with torch.no_grad():
                r_p = lambda_proj(r)
                L_truth = prob(x.numpy(), r_p.numpy())
                data_iteration.append(x, r_p, L_truth)
        lambda_optimizer.zero_grad()
        L_lambda.backward(retain_graph=True)
        lambda_optimizer.step()
        #x_optimizer.zero_grad()
        #L_x.backward()
        #x_optimizer.step()
        '''


        for param in L_model.parameters():
            param.requires_grad = True
        for _ in range(50):
            L_optimizer.zero_grad()

            id_sample = sample(range(len(data_iteration.x_past)),min(10,len(data_iteration.x_past)))
            
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
    num_iteration = 1000
    num_frame = 10

    x_model = x_LSTM(len_x, len_lambda, arg_nn)
    L_model = L_MLP(len_x, len_lambda, arg_nn)
    #L_model = L_LSTM(len_x, len_lambda, arg_nn)
    lambda_model = lambda_LSTM(len_lambda, arg_nn)
    x_optimizer = torch.optim.Adam(x_model.parameters(), lr=0.001)
    L_optimizer = torch.optim.Adam(L_model.parameters(), lr=0.001)
    lambda_optimizer = torch.optim.Adam(lambda_model.parameters(), lr=0.001)

    r = torch.randn(1,len_lambda)
    x = torch.randn(1,len_x)
    init_var = (x, r)

    model = (x_model, L_model, lambda_model)
    optimizer = (x_optimizer, L_optimizer, lambda_optimizer)
    
    my_train(L, init_var, model, num_iteration, num_frame, optimizer)

    print(out)
