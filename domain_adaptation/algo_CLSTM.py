import torch
import numpy as np
from nn_CLSTM_TL import *
from sys_prob_TL import problem_generator
from random import sample
from matplotlib import pyplot as plt

from data_set import data, r_data

torch.autograd.set_detect_anomaly(True)

np.random.seed(10)
torch.random.manual_seed(10000)



def derive_grad_lambda(problems, x, r):
    alpha = 2
    dr = torch.where(r < -1, -alpha, 0)
    dr = torch.where(r > 1, alpha, dr)
    dr = torch.where((r >= -1) & (r <= 1), alpha*r**(alpha-1), dr)
    partial_grad_lambda_p = torch.tensor([problems[i].gradient_lambda(x[:,i,:].numpy()) for i in range(len(problems))],dtype=torch.float32)
    grad_lambda = dr * partial_grad_lambda_p.transpose(0,1)
    return grad_lambda

def derive_grad_x(problems, x, r):
    grad_x = torch.tensor([problems[i].gradient_x(x[:,i,:].numpy(), r[:,i,:].numpy()) for i in range(len(problems))],dtype=torch.float32)
    return grad_x.transpose(0,1)
    



def my_train_true_gradient(problems, model, num_epoch, num_frame, num_iteration, optimizer):
 
 
    (x_optimizer, lambda_optimizer) = optimizer
    (x_model, lambda_model) = model

    obj_truth_result = []
    acc = []
    for n_p, prob in enumerate(problems):
        obj_truth_result.append(prob.objective(prob.solve().x.reshape(1,-1)))
    

    for epoch in range(num_epoch):

        # randomly initialize x,r and hidden states
        init_x = torch.randn(1,len(problems),len_x)
        init_r = torch.randn(1,len(problems),len_lambda)

        hidden_x = (torch.randn(1, len(problems), arg_nn.hidden_size), torch.randn(1, len(problems), arg_nn.hidden_size))
        hidden_lambda = (torch.randn(1, len(problems), arg_nn.hidden_size), torch.randn(1, len(problems), arg_nn.hidden_size))


        for param in x_model.parameters():
            param.requires_grad = True
        for param in lambda_model.parameters():
            param.requires_grad = True
        reserved_r = init_r
        reserved_x = init_x

        for frame in range(num_frame):
       
            for iter in range(num_iteration):
                r = reserved_r.detach()
                x = reserved_x.detach()

                grad_lambda = derive_grad_lambda(problems, x, r)
                delta_lambda, hidden_lambda = lambda_model(grad_lambda, h_s=hidden_lambda)

                hidden_lambda = (hidden_lambda[0].detach(), hidden_lambda[1].detach())
                _r = r + delta_lambda
                r = _r.detach()
                grad_lambda = derive_grad_lambda(problems, x, r)
                #_r.backward(-grad_lambda, retain_graph=True)

                r_p = lambda_proj(r)
                grad_x = derive_grad_x(problems, x, r_p)
                #grad_x = torch.tensor(prob.gradient_x(x.detach().numpy(), r_p.detach().numpy()),dtype=torch.float32)
                delta_x,hidden_x = x_model(grad_x, h_s=hidden_x)
                hidden_x = (hidden_x[0].detach(), hidden_x[1].detach())
                # hidden state x initialized
                #hidden_x = None
                _x = x + delta_x
                x = _x.detach()
                grad_x = derive_grad_x(problems, x, r_p)
                #grad_x = torch.tensor(prob.gradient_x(x.detach().numpy(), r_p.detach().numpy()),dtype=torch.float32)
                #_x.backward(grad_x, retain_graph=True)

                reserved_r = r
                reserved_x = x
            



                        
            #lambda_optimizer.step()
            #lambda_optimizer.zero_grad()
            #x_optimizer.step()
            #x_optimizer.zero_grad()
            precision  = 0
            for n_p, prob in enumerate(problems):
                r_p = lambda_proj(reserved_r[:,n_p,:])
                x = reserved_x[:,n_p,:]
                precision += 1 - np.abs(prob.objective(x.detach().numpy())-obj_truth_result[n_p])/np.abs(obj_truth_result[n_p])
            precision /= len(problems)
            acc.append(precision)
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
        # Save models and accuracy array
       
        # print result each iteration
        print("lambda:",r_p.detach().numpy())
        print("x:",x.detach().numpy())
        res = prob.solve()
        print("opt x: ", res.x)
        #print("constraint function: ", prob.check_con(_x.detach().numpy())[-1])
        #print("opt obj: ", prob.objective(res.x.reshape(1,-1)))
            
    #torch.save(x_model.state_dict(), 'x_model.pth')
    #torch.save(lambda_model.state_dict(), 'lambda_model.pth')
    #np.save('accuracy.npy', np.array(acc))

    # end of iterations
    #np.save('L_train.npy',np.array(L_train_result))
    #np.save('Loss_train.npy',np.array(Loss_train_result))
    #np.save('L_truth.npy',np.array(L_truth_result))
    #np.save('obj_train.npy',np.array(obj_truth_result))



if __name__ == "__main__":

    class prob_arg_source:
        sigma_1 = 1
        mu_1 = 0
        sigma_2 = 1
        mu_2 = -5
        ub = 10
        total_resource = 15

    

    problems = [problem_generator(prob_arg_source) for _ in range(2000)]
   
    class arg_nn:
        hidden_size = 32
        hidden_size_x = 20
    len_x = 5
    len_lambda = 2 * len_x +1

    num_epoch = 10
    num_iteration = 30
    num_frame = 900//30
    
    

    x_model = x_LSTM(len_x, arg_nn)
    lambda_model = lambda_LSTM(len_lambda, arg_nn)
    x_model.load_state_dict(torch.load('x_model.pth'))
    lambda_model.load_state_dict(torch.load('lambda_model.pth'))
    x_optimizer = torch.optim.Adam(x_model.parameters(), lr=0.001)
    lambda_optimizer = torch.optim.Adam(lambda_model.parameters(), lr=0.0005)

    model = (x_model, lambda_model)
    optimizer = (x_optimizer, lambda_optimizer)
    
    my_train_true_gradient(problems, model, num_epoch, num_frame, num_iteration, optimizer)

