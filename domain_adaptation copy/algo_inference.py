import torch
import numpy as np
from nn_CLSTM_TL import *
from sys_prob_TL import problem_generator
#torch.autograd.set_detect_anomaly(True)

np.random.seed(10000)
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
    
def D_step(model, D_loss,problems, x, r, hidden_state, is_source = True, is_lambda = True, is_train = True):
    (x_model, lambda_model, D_model_x, D_model_lambda) = model
    if is_lambda:
        grad = derive_grad_lambda(problems, x, r)
        model = lambda_model 
        D_model = D_model_lambda
    else:
        r_p = lambda_proj(r)
        grad = derive_grad_x(problems, x, r_p.detach())
        model = x_model
        D_model = D_model_x
    delta, hidden_state = model(grad, h_s = hidden_state)
    hidden_state = (hidden_state[0].detach(), hidden_state[1].detach())
    if not is_train:
        return delta, hidden_state
    temp = D_model(hidden_state[1])
    if is_source:
        loss = 1 * D_loss(temp,torch.ones_like(temp))
    else:
        loss = D_loss(temp,torch.zeros_like(temp))
    loss.backward(retain_graph=True)
    return loss, delta, hidden_state
    
 

def my_eval(problems, model, num_epoch, num_iteration):
 
    (x_model, lambda_model, D_model_x, D_model_lambda) = model
    (prob_source, prob_target) = problems
    D_loss = nn.BCELoss()
    obj_truth_result = {'source':[], 'target':[]}

    acc = {'source':[], 'target':[]}


    # to store the solution to source problems
    for n_p, prob in enumerate(prob_source):
        obj_truth_result['source'].append(prob.objective(prob.solve().x.reshape(1,-1)))
    # to store the solution to target problems
    for n_p, prob in enumerate(prob_target):
        obj_truth_result['target'].append(prob.objective(prob.solve().x.reshape(1,-1)))

    x_model.eval()
    lambda_model.eval()
    D_model_x.eval()
    D_model_lambda.eval()
    
    init_x_s = torch.randn(1, len(prob_source),len_x)
    init_r_s = torch.randn(1, len(prob_source),len_lambda)
    init_x_t = torch.randn(1, len(prob_target),len_x)
    init_r_t = torch.randn(1, len(prob_target),len_lambda)

    hidden_x_s = (torch.randn(1, len(prob_source), arg_nn.hidden_size), torch.randn(1, len(prob_source), arg_nn.hidden_size))
    hidden_lambda_s = (torch.randn(1, len(prob_source), arg_nn.hidden_size), torch.randn(1, len(prob_source), arg_nn.hidden_size))
    hidden_x_t = (torch.randn(1, len(prob_target), arg_nn.hidden_size), torch.randn(1, len(prob_target), arg_nn.hidden_size))
    hidden_lambda_t = (torch.randn(1, len(prob_target), arg_nn.hidden_size), torch.randn(1, len(prob_target), arg_nn.hidden_size))

    reserved_x_s = init_x_s
    reserved_r_s = init_r_s
    reserved_x_t = init_x_t
    reserved_r_t = init_r_t
    for epoch in range(num_epoch):

        # randomly initialize x,r and hidden states
        


        for iter in range(num_iteration):

            r_s = reserved_r_s.detach()
            x_s = reserved_x_s.detach()
            r_t = reserved_r_t.detach()
            x_t = reserved_x_t.detach()


            delta_lambda_s, hidden_lambda_s = D_step(model, D_loss, prob_source, x_s, r_s, hidden_lambda_s, is_source = True, is_lambda = True, is_train = False)
            delta_lambda_t, hidden_lambda_t = D_step(model, D_loss, prob_target, x_t, r_t, hidden_lambda_t, is_source = False, is_lambda = True, is_train = False)

    
            _r_s = r_s + delta_lambda_s
            r_s = _r_s.detach()
            _r_t = r_t + delta_lambda_t
            r_t = _r_t.detach()

            
            
            delta_x_s, hidden_x_s = D_step(model, D_loss, prob_source, x_s, r_s, hidden_x_s, is_source = True, is_lambda = False, is_train = False)
            delta_x_t, hidden_x_t = D_step(model, D_loss, prob_target, x_t, r_t, hidden_x_t, is_source = False, is_lambda = False, is_train = False)
            
            
            _x_s = x_s + delta_x_s
            x_s = _x_s.detach()
            _x_t = x_t + delta_x_t
            x_t = _x_t.detach()
            
            reserved_r_s = r_s
            reserved_x_s = x_s
            reserved_r_t = r_t
            reserved_x_t = x_t
            

            precision_s  = 0
            for n_p, prob in enumerate(prob_source):
                r_p = lambda_proj(reserved_r_s[:,n_p,:])
                x = reserved_x_s[:,n_p,:]
                precision_s += 1 - np.abs(prob.objective(x.detach().numpy())-obj_truth_result['source'][n_p])/np.abs(obj_truth_result['source'][n_p])
            precision_s /= len(prob_source)
            acc['source'].append(precision_s)

            precision_t  = 0
            for n_p, prob in enumerate(prob_target):
                r_p = lambda_proj(reserved_r_s[:,n_p,:])
                x = reserved_x_t[:,n_p,:]
                precision_t += 1 - np.abs(prob.objective(x.detach().numpy())-obj_truth_result['target'][n_p])/np.abs(obj_truth_result['target'][n_p])
            precision_t /= len(prob_target)
            acc['target'].append(precision_t)
            print("precision_source: ", 'source', precision_s, 'target', precision_t)
        np.save('acc_source_TL.npy', acc['source'])
        np.save('acc_target_TL.npy', acc['target'])



if __name__ == "__main__":

    class prob_arg_source:
        sigma_1 = 1
        mu_1 = 0
        sigma_2 = 1
        mu_2 = -5
        ub = 10
        total_resource = 15

    class prob_arg_target:
        sigma_1 = 2
        mu_1 = 3
        sigma_2 = 1
        mu_2 = -1
        ub = 15
        total_resource = 20

    class prob_arg_target:
        sigma_1 = 2
        mu_1 = -3
        sigma_2 = 1
        mu_2 = 2
        ub = 5
        total_resource = 20


    problems_source = [problem_generator(prob_arg_source) for _ in range(10000)]
    problems_target = [problem_generator(prob_arg_target) for _ in range(1000)]
    problems = (problems_source, problems_target)
   
    class arg_nn:
        hidden_size = 32
        hidden_size_x = 20
    len_x = 5
    len_lambda = 2 * len_x +1

    num_epoch = 10
    num_iteration = 30

    
    

    x_model = x_LSTM(len_x, arg_nn)
    lambda_model = lambda_LSTM(len_lambda, arg_nn)
    D_model_x = Discriminator(32, arg_nn)
    D_model_lambda = Discriminator(32, arg_nn)

    x_model.load_state_dict(torch.load('x_model_target.pth'))
    lambda_model.load_state_dict(torch.load('lambda_model_target.pth'))

    model = (x_model, lambda_model, D_model_x, D_model_lambda)
    
    
    my_eval(problems, model, num_epoch,num_iteration)