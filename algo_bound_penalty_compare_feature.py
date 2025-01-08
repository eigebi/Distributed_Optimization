import torch
import numpy as np
from nn_bound_compare import *
from sys_prob_penalty_compare import problem_generator
from random import sample
from matplotlib import pyplot as plt

from data_set import data, r_data

torch.autograd.set_detect_anomaly(True)

np.random.seed(10000)
torch.random.manual_seed(10000)



def derive_grad_lambda(prob, x_model, r):
    r.requires_grad = True
    r_p = lambda_proj(r)
    _x = x_model(r_p, torch.tensor(prob.feature, dtype=torch.float32))

    partial_grad_x = torch.tensor(prob.gradient_x(_x.detach().numpy(), r_p.detach().numpy()),dtype=torch.float32)
    partial_grad_lambda_p = torch.tensor(prob.gradient_lambda(_x.detach().numpy()),dtype=torch.float32)
    r_p.backward(partial_grad_lambda_p, retain_graph=True)
    _x.backward(partial_grad_x)
    grad_lambda = r.grad
  
    return grad_lambda

    



def my_train_true_gradient(problems, model, num_epoch, num_frame, num_iteration, optimizer):
 
 
    (x_optimizer, lambda_optimizer) = optimizer
    (x_model, lambda_model, x_copy) = model
    
    data_iteration = data()
    x_data_iteration = [r_data() for _ in range(len(problems))]
    L_train_result = []
    Loss_train_result = []
    L_truth_result = []
    obj_truth_result = []
    for n_p, prob in enumerate(problems):
        obj_truth_result.append(prob.objective(prob.solve().x.reshape(1,-1)))
    

    
    for epoch in range(num_epoch):
        
        # randomly initialize x,r and hidden states
        init_x = torch.randn(len(problems),len_x)
        init_r = torch.randn(len(problems),len_lambda)

        
        hidden_lambda = (torch.randn(1, arg_nn.hidden_size), torch.randn(1, arg_nn.hidden_size))
        #hidden_x = (torch.randn(1, arg_nn.hidden_size), torch.randn(1, arg_nn.hidden_size))

        for param in x_model.parameters():
            param.requires_grad = True
        for param in lambda_model.parameters():
            param.requires_grad = True
        for param in x_copy.parameters():
            param.requires_grad = False

        reserved_r = init_r
        reserved_x = init_x


        for frame in range(num_frame):
            
            
        
            for n_p, prob in enumerate(problems):
                
                for iter in range(num_iteration):

                    r = reserved_r[n_p].view(1,-1).detach()
                    x = reserved_x[n_p].view(1,-1).detach()

                    x_copy.load_state_dict(x_model.state_dict())
                    grad_lambda = derive_grad_lambda(prob, x_copy, r)
                    delta_lambda = lambda_model(grad_lambda, h_s=hidden_lambda)
                    hidden_lambda = None
                    r = r.detach()
                    _r = r + delta_lambda
                    r = _r.detach()
                    grad_lambda = derive_grad_lambda(prob, x_copy, r)
                    _r.backward(-grad_lambda, retain_graph=True)
                    reserved_r[n_p] = r.view(-1)
                    x_data_iteration[n_p].append(lambda_proj(r+np.random.randn()*0.1))
                
                _x = x_copy(r, torch.tensor(prob.feature, dtype=torch.float32))
                reserved_x[n_p] = _x.view(-1).detach()

            lambda_optimizer.step()
            lambda_optimizer.zero_grad()
            
            
            

            for iter in range(50):
                for n_p, prob in enumerate(problems):
                    sampled_id = sample(range(len(x_data_iteration[n_p].r_p)),min(5,len(x_data_iteration[n_p].r_p)))
                    r_p_data = torch.tensor(x_data_iteration[n_p].r_p,dtype=torch.float32)[sampled_id]
                    feature_data = torch.tensor(prob.feature, dtype=torch.float32).repeat(r_p_data.size(0), 1)
                    _x = x_model(r_p_data, feature_data)
                    grad_x = torch.tensor(prob.gradient_x(_x.detach().numpy(), r_p_data.detach().numpy()), dtype=torch.float32)
                    _x.backward(grad_x, retain_graph=True)

                
                x_optimizer.step()
                x_optimizer.zero_grad()


            precision  = 0
            for n_p, prob in enumerate(problems):
                r_p = lambda_proj(reserved_r[n_p].view(1,-1))
                x = reserved_x[n_p].view(1,-1)
                precision += 1 - np.abs(prob.objective(x.detach().numpy())-obj_truth_result[n_p])/np.abs(obj_truth_result[n_p])
            precision /= len(problems)
            print("precision: ", precision)

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
            




           

    


    # end of iterations
    
    np.save('L_truth_non_convex_p.npy',np.array(L_truth_result))
    np.save('obj_train_non_convex_p.npy',np.array(obj_truth_result))



if __name__ == "__main__":

    
    L = problem_generator(bounded=True)
    result = L.solve()
    out = result.x
    obj = result.fun


    problems = [problem_generator(bounded=True) for _ in range(1)]
   
    class arg_nn:
        hidden_size = 32
        hidden_size_x = 32
        u_b = L.u_b
        # to be modified

    len_x = L.num_o
    len_lambda = 1
    num_epoch = 50
    num_iteration = 20
    num_frame = 200//20

    x_model = x_MLP(len_x, len_lambda, problems[0].feature.shape[0], arg_nn,bounded = True)
    x_copy = x_MLP(len_x, len_lambda, problems[0].feature.shape[0], arg_nn,bounded = True)
    x_copy.load_state_dict(x_model.state_dict())
    lambda_model = lambda_LSTM(len_lambda, arg_nn)
    x_optimizer = torch.optim.Adam(x_model.parameters(), lr=0.001)
    lambda_optimizer = torch.optim.Adam(lambda_model.parameters(), lr=0.002)


    model = (x_model, lambda_model, x_copy)
    optimizer = (x_optimizer, lambda_optimizer)
    
    my_train_true_gradient(problems, model, num_epoch, num_frame, num_iteration, optimizer)


    print(out)
