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
    L_train_result = []
    Loss_train_result = []
    L_truth_result = []
    

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
            for _ in range(10):
                _x = x_model(r_p)
                L = L_model(_x, r_p)
                # store one piece of data after some variable is updated
                L_truth = prob(_x.detach().numpy(), r_p.detach().numpy())
                
                L_train_result.append(L.detach().numpy())
                L_truth_result.append(L_truth)

                data_iteration.append(_x, r_p, L_truth)

                #L_x = L_x + L
                # derive the gradient w.r.t. r
                x_optimizer.zero_grad()
                L.backward()
                x_optimizer.step()
                #x = _x.detach()
                #print("L_x:",L.detach().numpy())

            for param in x_model.parameters():
                param.requires_grad = False
            r.requires_grad = True
            r_p = lambda_proj(r)
            _x = x_model(r_p)
            L = L_model(_x, r_p)
            
            L.backward()
            grad_lambda = r.grad

            L_truth = prob(_x.detach().numpy(), r_p.detach().numpy())
            data_iteration.append(_x, r_p, L_truth)


            # update r
            # current r
            r = r.detach()
            #x = _x.detach()
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
            _x = x_model(r_p)
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
            _x = x_model(r_p)

            L_truth = prob(_x.detach().numpy(), r_p.detach().numpy())
            data_iteration.append(_x, r_p, L_truth)


            x = _x.detach()
            r = _r.detach()


        for param in L_model.parameters():
            param.requires_grad = True
        temp_loss = 0
        for _ in range(50):
            L_optimizer.zero_grad()

            id_sample = sample(range(len(data_iteration.x_past)),min(10,len(data_iteration.x_past)))
            
            x_data = torch.tensor(data_iteration.x_past,dtype=torch.float32)[id_sample]
            r_data = torch.tensor(data_iteration.lambda_past,dtype=torch.float32)[id_sample]
            L_truth = torch.tensor(data_iteration.L_past,dtype=torch.float32)[id_sample]

            loss_L = loss_MSE(L_model(x_data.view(-1,len_x),r_data.view(-1,len_lambda)),L_truth.view(-1,1))
            loss_L.backward()
            L_optimizer.step()
            temp_loss += loss_L.detach().numpy()
        Loss_train_result.append(temp_loss)
        

        print("L loss", loss_L.detach().numpy(),'delta',out-x[0].detach().numpy())

        # here we update L model

    print("lambda:",r_p.detach().numpy())
    print("x:",x.detach().numpy())
    np.save('L_train.npy',np.array(L_train_result))
    np.save('Loss_train.npy',np.array(Loss_train_result))
    np.save('L_truth.npy',np.array(L_truth_result))


def my_train_true_gradient(prob, init_var ,model, num_iteration, num_frame, optimizer):
 
 
    (x_optimizer, L_optimizer, lambda_optimizer) = optimizer
    (x_model, L_model, lambda_model) = model
    (x, r) = init_var
    
    


    loss_MSE = torch.nn.MSELoss()
    data_iteration = data()
    L_train_result = []
    Loss_train_result = []
    L_truth_result = []
    obj_train_result = []
    

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
            for _ in range(15):
                _x = x_model(r_p)
                grad_x = torch.tensor(prob.gradient_x(_x.detach().numpy(), r_p.detach().numpy()),dtype=torch.float32)
                #L = prob(_x, r_p)
                # store one piece of data after some variable is updated
                L_truth = prob(_x.detach().numpy(), r_p.detach().numpy())
                obj_truth = prob.objective(_x.detach().numpy())
                obj_train_result.append(obj_truth)
                
                #L_train_result.append(L.detach().numpy())
                L_truth_result.append(L_truth)

                data_iteration.append(_x, r_p, L_truth)

                #L_x = L_x + L
                # derive the gradient w.r.t. r
                x_optimizer.zero_grad()
                _x.backward(grad_x)
                x_optimizer.step()
                #x = _x.detach()
                #print("L_x:",L.detach().numpy())

            for param in x_model.parameters():
                param.requires_grad = False
            r.requires_grad = True
            r_p = lambda_proj(r)
            _x = x_model(r_p)
            partial_grad_x = torch.tensor(prob.gradient_x(_x.detach().numpy(), r_p.detach().numpy()),dtype=torch.float32)
            partial_grad_lambda = torch.tensor(prob.gradient_lambda(_x.detach().numpy()),dtype=torch.float32)
            #L = L_model(_x, r_p)
            
            #L.backward()
            _x.backward(partial_grad_x)

            grad_lambda = r.grad + partial_grad_lambda

            L_truth = prob(_x.detach().numpy(), r_p.detach().numpy())
            data_iteration.append(_x, r_p, L_truth)


            # update r
            # current r
            r = r.detach()
            #x = _x.detach()
            for param in x_model.parameters():
                param.requires_grad = False
            for param in lambda_model.parameters():
                param.requires_grad = True
            
            # derive new r based on current r and current gradient w.r.t. r
            delta_lambda = lambda_model(grad_lambda)
            _r = r + delta_lambda
            #_r.retain_grad()
            
            
            r_p = lambda_proj(_r)
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


        for param in L_model.parameters():
            param.requires_grad = True
        temp_loss = 0
        for _ in range(1):
            L_optimizer.zero_grad()

            id_sample = sample(range(len(data_iteration.x_past)),min(10,len(data_iteration.x_past)))
            
            x_data = torch.tensor(data_iteration.x_past,dtype=torch.float32)[id_sample]
            r_data = torch.tensor(data_iteration.lambda_past,dtype=torch.float32)[id_sample]
            L_truth = torch.tensor(data_iteration.L_past,dtype=torch.float32)[id_sample]

            loss_L = loss_MSE(L_model(x_data.view(-1,len_x),r_data.view(-1,len_lambda)),L_truth.view(-1,1))
            loss_L.backward()
            L_optimizer.step()
            temp_loss += loss_L.detach().numpy()
        Loss_train_result.append(temp_loss)
        

        print("L loss", loss_L.detach().numpy(),'delta',out-x[0].detach().numpy())

        # here we update L model

    print("lambda:",r_p.detach().numpy())
    print("x:",x.detach().numpy())
    np.save('L_train.npy',np.array(L_train_result))
    np.save('Loss_train.npy',np.array(Loss_train_result))
    np.save('L_truth.npy',np.array(L_truth_result))
    np.save('obj_train.npy',np.array(obj_train_result))



if __name__ == "__main__":

    
    L = problem_generator()
    out = L.solve().x
   
    class arg_nn:
        hidden_size = 32
        hidden_size_x = 32
    len_x = 5
    len_lambda = 2 * len_x +1
    num_iteration = 500
    num_frame = 50

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
    
    #my_train(L, init_var, model, num_iteration, num_frame, optimizer)
    my_train_true_gradient(L, init_var, model, num_iteration, num_frame, optimizer)


    print(out)
