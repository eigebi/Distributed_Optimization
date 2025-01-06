import torch
import numpy as np
from nn_CLSTM import *
from sys_prob_CLSTM import problem_generator
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

    



def my_train_true_gradient(problems, model, num_epoch, num_frame, num_iteration, optimizer):
 
 
    (x_optimizer, lambda_optimizer) = optimizer
    (x_model, lambda_model) = model

    
    data_iteration = data()
    x_data_iteration = r_data()
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

        hidden_x = (torch.randn(1, arg_nn.hidden_size), torch.randn(1, arg_nn.hidden_size))
        hidden_lambda = (torch.randn(1, arg_nn.hidden_size), torch.randn(1, arg_nn.hidden_size))


        for param in x_model.parameters():
            param.requires_grad = True
        for param in lambda_model.parameters():
            param.requires_grad = True
        reserved_r = init_r
        reserved_x = init_x

        for frame in range(num_frame):
           

            for n_p, prob in enumerate(problems):
                #print("epoch: ", epoch, "frame: ", frame, "problem: ", n_p)

                for iter in range(num_iteration):
                    r = reserved_r[n_p].view(1,-1).detach()
                    x = reserved_x[n_p].view(1,-1).detach()

                    grad_lambda = derive_grad_lambda(prob, x, r)
                    delta_lambda = lambda_model(grad_lambda, h_s=hidden_lambda)
                    # hidden state lambda initialized
                    hidden_lambda = None
                    _r = r + delta_lambda
                    r = _r.detach()
                    grad_lambda = derive_grad_lambda(prob, x, r)
                    _r.backward(-grad_lambda, retain_graph=True)

                    r_p = lambda_proj(r)
                    grad_x = torch.tensor(prob.gradient_x(x.detach().numpy(), r_p.detach().numpy()),dtype=torch.float32)
                    delta_x = x_model(grad_x, h_s=hidden_x)
                    # hidden state x initialized
                    hidden_x = None
                    _x = x + delta_x
                    x = _x.detach()
                    grad_x = torch.tensor(prob.gradient_x(x.detach().numpy(), r_p.detach().numpy()),dtype=torch.float32)
                    _x.backward(grad_x, retain_graph=True)

                    reserved_r[n_p] = r.view(-1)
                    reserved_x[n_p] = x.view(-1)
            



                        
            lambda_optimizer.step()
            lambda_optimizer.zero_grad()
            x_optimizer.step()
            x_optimizer.zero_grad()
            precision  = 0
            for n_p, prob in enumerate(problems):
                r_p = lambda_proj(reserved_r[n_p].view(1,-1))
                x = reserved_x[n_p].view(1,-1)
                precision += 1 - np.abs(prob.objective(x.detach().numpy())-obj_truth_result[n_p])/np.abs(obj_truth_result[n_p])
            precision /= len(problems)
            print("precision: ", precision)
            #print("delta: ",  [problems[i].objective(reserved_x[i].view(1,-1).detach().numpy()) for i in range(len(problems))], obj_truth_result)
            # r_p = lambda_proj(r)

            #print("L truth: ", prob(x.detach().numpy(), r_p.detach().numpy()), "obj truth: ", prob.objective(x.detach().numpy()))
            #latest_r = r
            #latest_x = x
            #r = reserved_r
            #x = reserved_x
                    
        #r = latest_r
        #x = latest_x

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

    problems = [problem_generator() for _ in range(20)]
   
    class arg_nn:
        hidden_size = 20
        hidden_size_x = 20
    len_x = 10
    len_lambda = 2 * len_x +1

    num_epoch = 50
    num_iteration = 5
    num_frame = 500//5
    
    

    x_model = x_LSTM(len_x, arg_nn)
    lambda_model = lambda_LSTM(len_lambda, arg_nn)
    x_optimizer = torch.optim.Adam(x_model.parameters(), lr=0.004)
    lambda_optimizer = torch.optim.Adam(lambda_model.parameters(), lr=0.001)

    model = (x_model, lambda_model)
    optimizer = (x_optimizer, lambda_optimizer)
    
    my_train_true_gradient(problems, model, num_epoch, num_frame, num_iteration, optimizer)


    print(out)
