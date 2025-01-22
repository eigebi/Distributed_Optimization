import torch
import numpy as np
from nn_CLSTM_TL import *
from sys_prob_TL import problem_generator
#torch.autograd.set_detect_anomaly(True)

np.random.seed(10009)
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
    
    


def train_x_lambda(x_model, lambda_model, x_optimizer, lambda_optimizer, D_model_x, D_model_lambda, D_optimizer_x, D_optimizer_lambda, D_loss, hidden_x, hidden_lambda, data_x, data_lambda, label_x, label_lambda):
    pass



def my_train_true_gradient(problems, model, num_epoch, num_frame, num_iteration, optimizer):
 
    (x_optimizer, lambda_optimizer, D_optimizer_x, D_optimizer_lambda) = optimizer
    (x_model, lambda_model, D_model_x, D_model_lambda) = model
    (prob_source, prob_target) = problems
    D_loss = nn.BCELoss()
    obj_truth_result = {'source':[], 'target':[]}
    acc = []


    # to store the solution to source problems
    for n_p, prob in enumerate(prob_source):
        obj_truth_result['source'].append(prob.objective(prob.solve().x.reshape(1,-1)))
    # to store the solution to target problems
    for n_p, prob in enumerate(prob_target):
        obj_truth_result['target'].append(prob.objective(prob.solve().x.reshape(1,-1)))
    

    for epoch in range(num_epoch):

        # randomly initialize x,r and hidden states
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


        for param in x_model.parameters():
            param.requires_grad = False
        for param in lambda_model.parameters():
            param.requires_grad = False
        for param in D_model_x.parameters():
            param.requires_grad = True
        for param in D_model_lambda.parameters():
            param.requires_grad = True

        # train Discriminator
        for frame in range(5):
            '''
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
            '''
            
            for iter in range(30):

                # generate piecese of data for D to learn
                r_s = reserved_r_s.detach()
                x_s = reserved_x_s.detach()
                r_t = reserved_r_t.detach()
                x_t = reserved_x_t.detach()

                D_optimizer_lambda.zero_grad()
                loss_lambda_s, delta_lambda_s, hidden_lambda_s = D_step(model, D_loss, prob_source, x_s, r_s, hidden_lambda_s, is_source = True, is_lambda = True, is_train = True)
                loss_lambda_t, delta_lambda_t, hidden_lambda_t = D_step(model, D_loss, prob_target, x_t, r_t, hidden_lambda_t, is_source = False, is_lambda = True, is_train = True)
                D_optimizer_lambda.step()
                

                
                _r_s = r_s + delta_lambda_s
                r_s = _r_s.detach()
                _r_t = r_t + delta_lambda_t
                r_t = _r_t.detach()

                
                D_optimizer_x.zero_grad()
                loss_x_s, delta_x_s, hidden_x_s = D_step(model, D_loss, prob_source, x_s, r_s, hidden_x_s, is_source = True, is_lambda = False, is_train = True)
                loss_x_t, delta_x_t, hidden_x_t = D_step(model, D_loss, prob_target, x_t, r_t, hidden_x_t, is_source = False, is_lambda = False, is_train = True)
                D_optimizer_x.step()
              
                _x_s = x_s + delta_x_s
                x_s = _x_s.detach()
                _x_t = x_t + delta_x_t
                x_t = _x_t.detach()
                
                reserved_r_s = r_s
                reserved_x_s = x_s
                reserved_r_t = r_t
                reserved_x_t = x_t
            

            precision  = 0
            for n_p, prob in enumerate(prob_source):
                r_p = lambda_proj(reserved_r_s[:,n_p,:])
                x = reserved_x_s[:,n_p,:]
                precision += 1 - np.abs(prob.objective(x.detach().numpy())-obj_truth_result['source'][n_p])/np.abs(obj_truth_result['source'][n_p])
            precision /= len(prob_source)
            acc.append(precision)
            print("precision_source: ", precision)

            print("loss: ", "loss_x_s", loss_x_s.detach().numpy(),"loss_x_t", loss_x_t.detach().numpy(), "loss_lambda_s", loss_lambda_s.detach().numpy(), "loss_lambda_t", loss_lambda_t.detach().numpy())
        


        for param in x_model.parameters():
            param.requires_grad = True
        for param in lambda_model.parameters():
            param.requires_grad = True
        for param in D_model_x.parameters():
            param.requires_grad = False
        for param in D_model_lambda.parameters():
            param.requires_grad = False
        
        for frame in range(5):
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
            for _ in range(20):
                for iter in range(30):

                    # train x and lambda
                    
                    r_t = reserved_r_t.detach()
                    x_t = reserved_x_t.detach()

                    
                    grad_lambda_t = derive_grad_lambda(prob_target, x_t, r_t)

                    delta_lambda_t, hidden_lambda_t = lambda_model(grad_lambda_t, h_s=hidden_lambda_t)
                    temp = D_model_lambda(hidden_lambda_t[1])
                    hidden_lambda_t = (hidden_lambda_t[0].detach(), hidden_lambda_t[1].detach())
                    _r_t = r_t + delta_lambda_t
                    r_t = _r_t.detach()
                    grad_lambda_t = derive_grad_lambda(prob_target, x_t, r_t)
                    #混淆用标签
                    loss_lambda = 20*D_loss(temp,torch.ones_like(temp))
                    loss_lambda.backward(retain_graph=True)
                    _r_t.backward(-grad_lambda_t, retain_graph=True)
                    r_p_t = lambda_proj(r_t)


                    grad_x_t = derive_grad_x(prob_target, x_t, r_p_t)
                    delta_x_t, hidden_x_t = x_model(grad_x_t, h_s=hidden_x_t)
                    temp = D_model_x(hidden_x_t[1])
                    hidden_x_t = (hidden_x_t[0].detach(), hidden_x_t[1].detach())
                    _x_t = x_t + delta_x_t
                    x_t = _x_t.detach()
                    grad_x_t = derive_grad_x(prob_target, x_t, r_p_t)
                    loss_x = 20*D_loss(temp,torch.ones_like(temp))
                    loss_x.backward(retain_graph=True)
                    _x_t.backward(grad_x_t, retain_graph=True)

                    

                    reserved_r_s = r_s
                    reserved_x_s = x_s
                    reserved_r_t = r_t
                    reserved_x_t = x_t
                lambda_optimizer.step()
                lambda_optimizer.zero_grad()
                x_optimizer.step()
                x_optimizer.zero_grad()
                precision  = 0
                for n_p, prob in enumerate(prob_target):
                    r_p = lambda_proj(reserved_r_t[:,n_p,:])
                    x = reserved_x_t[:,n_p,:]
                    precision += 1 - np.abs(prob.objective(x.detach().numpy())-obj_truth_result['target'][n_p])/np.abs(obj_truth_result['target'][n_p])
                precision /= len(prob_target)
                acc.append(precision)
                print("precision_target: ", precision)
                print("loss: ", loss_x.detach().numpy(), loss_lambda.detach().numpy())
            torch.save(x_model.state_dict(), 'x_model_target.pth')
            torch.save(lambda_model.state_dict(), 'lambda_model_target.pth')
            torch.save(D_model_x.state_dict(), 'D_model_x.pth')
            torch.save(D_model_x.state_dict(), 'D_model_x.pth')
            np.save('accuracy_target.npy', np.array(acc))




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


    problems_source = [problem_generator(prob_arg_source) for _ in range(2000)]
    problems_target = [problem_generator(prob_arg_target) for _ in range(500)]
    problems = (problems_source, problems_target)
   
    class arg_nn:
        hidden_size = 32
        hidden_size_x = 20
    len_x = 5
    len_lambda = 2 * len_x +1

    num_epoch = 10
    num_iteration = 5
    num_frame = 2
    
    

    x_model = x_LSTM(len_x, arg_nn)
    lambda_model = lambda_LSTM(len_lambda, arg_nn)
    D_model_x = Discriminator(32, arg_nn)
    D_model_lambda = Discriminator(32, arg_nn)

    x_model.load_state_dict(torch.load('x_model_target.pth'))
    lambda_model.load_state_dict(torch.load('lambda_model_target.pth'))

    x_optimizer = torch.optim.Adam(x_model.parameters(), lr=0.001, betas=(0.5,0.999))
    lambda_optimizer = torch.optim.Adam(lambda_model.parameters(), lr=0.001, betas=(0.5,0.999))
    D_optimizer_x = torch.optim.Adam(D_model_x.parameters(), lr=0.001, betas=(0.5,0.999))
    D_optimizer_lambda = torch.optim.Adam(D_model_lambda.parameters(), lr=0.001, betas=(0.5,0.999))

    model = (x_model, lambda_model, D_model_x, D_model_lambda)
    optimizer = (x_optimizer, lambda_optimizer, D_optimizer_x, D_optimizer_lambda)
    
    my_train_true_gradient(problems, model, num_epoch, num_frame, num_iteration, optimizer)