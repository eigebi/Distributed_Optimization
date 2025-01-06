import torch
import numpy as np
from nn_bound_compare import *
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

    



def my_train_true_gradient(problems, model, num_epoch, num_frame, num_iteration, optimizer):
 
 
    (x_optimizer, lambda_optimizer) = optimizer
    (x_model, lambda_model) = model
    
    data_iteration = data()
    x_data_iteration = [r_data() for _ in range(len(problems))]
    L_train_result = []
    Loss_train_result = []
    L_truth_result = []
    obj_truth_result = []
    
    # there is only 1 initialization for x_LSTM
    hidden_x = (torch.randn(1, arg_nn.hidden_size), torch.randn(1, arg_nn.hidden_size))
    for epoch in range(num_epoch):
        
        # randomly initialize x,r and hidden states
        init_x = torch.zeros(len(problems),len_x)
        init_r = torch.randn(len(problems),len_lambda)

        
        hidden_lambda = (torch.randn(1, arg_nn.hidden_size), torch.randn(1, arg_nn.hidden_size))

        for param in x_model.parameters():
            param.requires_grad = False
        for param in lambda_model.parameters():
            param.requires_grad = True

        reserved_r = init_r
        reserved_x = init_x


        for frame in range(num_frame):
        
            for n_p, prob in enumerate(problems):
                
                for iter in range(num_iteration):

                    r = reserved_r[n_p].view(1,-1).detach()
                    
                    grad_lambda = derive_grad_lambda(prob, x_model, r)
                    r = r.detach()
                    delta_lambda = lambda_model(grad_lambda)
                    _r = r + delta_lambda
                    r = _r.detach()
                    grad_lambda = derive_grad_lambda(prob, x_model, r)
                    _r.backward(-grad_lambda, retain_graph=True)
                    # above is update for lambda only
                    reserved_r[n_p] = r.view(-1)
                

                    # to be modified, add problem idx
                    # record the lambda trajectory
                    x_data_iteration[n_p].append(lambda_proj(r))

                    
                    #r_p = lambda_proj(r)
                    #_x = x_model(r_p)
                #print("lambda: ", r_p.detach().numpy())
                    
                    

                    
            lambda_optimizer.step()
            lambda_optimizer.zero_grad()
                #x_optimizer.step()
                #x_optimizer.zero_grad()

            #r_p = lambda_proj(r)
            #_x = x_model(r_p)
            #print("L truth: ", prob(_x.detach().numpy(), r_p.detach().numpy()), "obj truth: ", prob.objective(_x.detach().numpy()))
            #latest_r = r
            #r = reserved_r
            #r = latest_r

            # print result each iteration
            #print("lambda:",r_p.detach().numpy())
            #print("x:",_x.detach().numpy())
            #res = prob.solve()
            #print("opt x: ", res.x)
            #print("constraint function: ", prob.check_con(_x.detach().numpy())[-1])
            #print("opt obj: ", prob.objective(res.x.reshape(1,-1)))
            




            # update x_model by samlping
            for param in x_model.parameters():
                param.requires_grad = True
            for param in lambda_model.parameters():
                param.requires_grad = False
            for n_p, prob in enumerate(problems):

                for iter in range(num_iteration):

                    x = reserved_x[n_p].view(1,-1).detach()
            
                    sampled_id = sample(range(len(x_data_iteration[n_p].r_p)),min(5,len(x_data_iteration[n_p].r_p)))
                    r_p_data = torch.tensor(x_data_iteration[n_p].r_p,dtype=torch.float32)[sampled_id]
                    _x = x_model(r_p_data)
                    grad_x = torch.tensor(prob.gradient_x_penalty(_x.detach().numpy(), r_p_data.detach().numpy()), dtype=torch.float32)
                    x_optimizer.zero_grad()
                    _x.backward(grad_x)
                    x_optimizer.step()
            r_p = lambda_proj(r)
            _x = x_model(r_p)
            print("L truth: ", prob(_x.detach().numpy(), r_p.detach().numpy()), "obj truth: ", prob.objective(_x.detach().numpy()))
            L_truth_result.append(prob(_x.detach().numpy(), r_p.detach().numpy()))
            obj_truth_result.append(prob.objective(_x.detach().numpy()))

    


    # end of iterations
    
    np.save('L_truth_non_convex_p.npy',np.array(L_truth_result))
    np.save('obj_train_non_convex_p.npy',np.array(obj_truth_result))



if __name__ == "__main__":

    
    L = problem_generator(bounded=True)
    result = L.solve()
    out = result.x
    obj = result.fun


    problems = [problem_generator(bounded=True) for _ in range(20)]
   
    class arg_nn:
        hidden_size = 32
        hidden_size_x = 32
        u_b = L.u_b
        # to be modified

    len_x = L.num_o
    len_lambda = 1
    num_epoch = 50
    num_iteration = 5
    num_frame = 500//5

    x_model = x_LSTM(len_x, len_lambda, arg_nn,bounded = True)
    lambda_model = lambda_LSTM(len_lambda, arg_nn)
    x_optimizer = torch.optim.Adam(x_model.parameters(), lr=0.001)
    lambda_optimizer = torch.optim.SGD(lambda_model.parameters(), lr=0.002)


    model = (x_model, lambda_model)
    optimizer = (x_optimizer, lambda_optimizer)
    
    my_train_true_gradient(problems, model, num_epoch, num_frame, num_iteration, optimizer)


    print(out)
