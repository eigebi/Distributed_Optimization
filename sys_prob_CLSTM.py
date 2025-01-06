import numpy as np
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
import scipy
np.random.seed(10086)

#test problem is simple and do not contains grouping or partitioning

#The base problem is a class that contains the objective function and the indices of variables that are involved in the function. The indices are drawn based on the global variables.
#The problem contains two operations, add and multiply. The computation result is to return the new optimization objective function, the new variable id, the graph among those variables, and the optimization direction (min/max). 
class BaseProblem:
    def __init__(self,func,varID,is_min=True):
        self.func = func
        #this ID is only used to call the global variable
        self.varID = np.array(varID)
        self.gamma = np.random.randn(len(varID))
        self.is_min = is_min
        self.x_next = []
    def __call__(self, x):
        return self.func(x)
    
    def __add__(self, other):
        return BaseProblem(lambda x: self.func(x) + other.func(x),self.varID)
    def __mul__(self, other):
        return BaseProblem(lambda x: self.func(x) * other.func(x),self.varID)
    def append(self):
        pass

    def optimize(self,x_global):
        rho = 1
        x_i = x_global[self.varID]
        f_i = lambda x: self.func(x)+self.gamma@(x-x_i)+rho/2*np.linalg.norm(x-x_i)**2
        if self.is_min:
            x = minimize(f_i,np.zeros(len(self.varID)))
        return



class prob:
    def __init__(self):
        self.obj = []
        self.con = []
        self.r = []


class problem_generator(prob):
    def __init__(self, bounded = False):
        super(problem_generator,self).__init__()
        num_o= 10
        self.num_o = num_o
        self.bounded = bounded
        f_s = []
        self.jac = []
        for _ in range(num_o):
            temp = np.random.randn(3,3)
            temp = temp @ temp.T
            temp2 = -np.random.randn(3)*2

            f_s.append(lambda x: x @ temp @ x + temp2 @ x+20)
            self.jac.append(lambda x: 2*temp @ x + temp2)

        for f in f_s:
            var_select = np.random.choice(num_o,3,replace=False)
            self.obj.append(BaseProblem(f,var_select))
        u_b = np.zeros(num_o+1)
        for i in range(num_o): 
            u_bound = np.random.randint(1,16)#16
            self.con.append(BaseProblem(lambda x: x - u_bound ,[i]))
            u_b[i]=u_bound
        for i in range(num_o):
            self.con.append(BaseProblem(lambda x: -x ,[i]))
        u_b[-1] = np.array([5],dtype=np.float32)
        self.con.append(BaseProblem(lambda x: np.array([np.sum(x)-u_b[-1]]) ,np.arange(num_o)))
        self.u_b = u_b
        
        
        for i in range(2*num_o+1):
            self.r.append(BaseProblem(lambda x: x , [i],is_min=False))

    def __call__(self,X,R):
        batch_size = X.shape[0]
        result = np.zeros(batch_size)
        if self.bounded:
            for i in range(batch_size):
                result[i] = np.sum([self.obj[k](X[i,self.obj[k].varID]) for k in range(self.num_o)]) + np.sum(np.multiply(R[i,:],np.array([np.array(self.con[-1](X[i,self.con[-1].varID]))])[0]))
        else:
            for i in range(batch_size):
                result[i] = np.sum([self.obj[k](X[i,self.obj[k].varID]) for k in range(self.num_o)]) + np.sum(np.multiply(R[i,:],np.array([np.array(self.con[k](X[i,self.con[k].varID])) for k in range(2*self.num_o+1)])[0]))
        return result.reshape(-1,1)
    
    def solve(self):
        f = lambda x: sum([self.obj[k](x[self.obj[k].varID]) for k in range(self.num_o)])   

        x0 = np.random.randn(self.num_o)
        A = np.row_stack((np.eye(self.num_o),np.ones((1,self.num_o))))
        constraint = LinearConstraint(A,0,self.u_b)
        res = minimize(f,x0,constraints=constraint)
        return res
    
    def gradient_x(self,x,r):
        grad_x = np.zeros([x.shape[0],self.num_o],dtype=np.float32)
        for k in range(x.shape[0]):
            for i in range(self.num_o):
                grad_x[k,self.obj[i].varID] += self.jac[i](x[k,self.obj[i].varID])
            if self.bounded:
                grad_x[k,:] = grad_x[k,:] + r[k,-1]
            else:
                grad_x[k,:] = grad_x[k,:] + r[k,:self.num_o] - r[k,self.num_o:-1] + r[k,-1]
        return grad_x
    
    def gradient_x_penalty(self,x,r):
        grad_x = np.zeros([x.shape[0],self.num_o],dtype=np.float32)
        for k in range(x.shape[0]):
            g = np.sum(x[k,:])
            penalty = 0
            if g>0:
                penalty = 5 * g
            
            for i in range(self.num_o):
                grad_x[k,self.obj[i].varID] += self.jac[i](x[k,self.obj[i].varID])
            if self.bounded:
                grad_x[k,:] = grad_x[k,:] + r[k,-1] + penalty
            else:
                grad_x[k,:] = grad_x[k,:] + r[k,:self.num_o] - r[k,self.num_o:-1] + r[k,-1]
        return grad_x
    
    def gradient_lambda(self,x):
        grad_lambda = np.zeros(2*self.num_o+1,dtype=np.float32)
        grad_lambda = np.array([self.con[k](x[0,self.con[k].varID]) for k in range(2*self.num_o+1)])
        if self.bounded:
            return grad_lambda[-1].reshape(1,-1)
        else:
            return grad_lambda.reshape(1,-1)
    
    def objective(self, x):
        return np.sum([self.obj[k](x[0,self.obj[k].varID]) for k in range(self.num_o)])
    
    def check_con(self,x):
        return np.array([self.con[k](x[0,self.con[k].varID]) for k in range(2*self.num_o+1)])
    






        

if __name__ == '__main__':


    num_o = 8


    f_s = []
    non_convex = True
    jac = []
    for _ in range(num_o):
        temp = np.random.randn(3,3)
        temp = temp @ temp.T
        temp2 = np.random.randn(3)*50

        f_s.append(lambda x: x @ temp @ x + temp2 @ x)
        jac.append(lambda x: 2*temp @ x + temp2)
        #generate 3 integers from (1,5) without repetition
        #np.random.choice(5,3,replace=False)
    p = prob()
    for f in f_s:
        var_select = np.random.choice(num_o,3,replace=False)
        p.obj.append(BaseProblem(f,var_select))
    u_b = np.zeros(num_o+1)
    for i in range(num_o): 
        u_bound = np.random.randint(1,16)
        p.con.append(BaseProblem(lambda x: x - u_bound ,[i]))
        u_b[i]=u_bound
    for i in range(num_o):
        p.con.append(BaseProblem(lambda x: -x ,[i]))
    p.con.append(BaseProblem(lambda x: np.sum(x)-30 ,np.arange(num_o)))
    u_b[-1] = 50
    for i in range(2*num_o+1):
        p.r.append(BaseProblem(lambda x: x , [i],is_min=False))
    # global x
    z = np.random.randn(num_o)
    # global lambda
    y = np.abs(np.random.randn(2*num_o+1))
    #gamma = np.abs(np.random.randn(11)) this gamma is local
    for i in range(num_o*2+1):
        p.r[i].x_next = y[i]


    T = 500
    rho = 2


    # need an index table to connect z to all the local variables, which is inverted to the varID
    # can be wrapped into a function
    z_obj_id = []
    z_con_id = []
    for i in range(num_o):
        temp_o = []
        temp_c = []
        for j in range(len(p.obj)):
            if i in p.obj[j].varID:
                id_ = np.where(p.obj[j].varID==i)
                temp_o.append([j,id_])
        z_obj_id.append(temp_o)
        for j in range(len(p.con)):
            if i in p.con[j].varID:
                id_ = np.where(p.con[j].varID==i)
                temp_c.append([j,id_])
        z_con_id.append(temp_c)


                
    for t in range(T):
        #update the objective function
        for i in range(num_o):
            # to be wrapped into optimize()
            z_i= z[p.obj[i].varID]
            if non_convex:
                f_i = lambda x: jac[i](x)@(x-z_i)+p.obj[i].gamma@(x-z_i)+rho/2*np.linalg.norm(x-z_i)**2
            else:
                f_i = lambda x: p.obj[i](x)+p.obj[i].gamma@(x-z_i)+rho/2*np.linalg.norm(x-z_i)**2
            x_i = minimize(f_i,np.zeros(len(p.obj[i].varID))).x
            p.obj[i].x_next = x_i
            #the square norm of x-z
        for i in range(2*num_o+1):
            # to be wrapped into optimize()
            z_i= z[p.con[i].varID]
            #f_i = lambda x: p.r[i].x_next*p.con[i].func(x)+p.con[i].gamma@(x-z_i)+rho/2*np.linalg.norm(x-z_i)**2
            f_i = lambda x: p.r[i].x_next*p.con[i](x)+p.con[i].gamma@(x-z_i)+rho/2*np.linalg.norm(x-z_i)**2
            x_i = minimize(f_i,np.zeros(len(p.con[i].varID))).x
            p.con[i].x_next = x_i
            #the square norm of x-y
            #it is better to store the con_func value to serve the next step update

        #update lambda in the lagrangian reformulation
        for i in range(num_o*2+1):
            # to be wrapped into optimize()
            y_i= y[p.r[i].varID]
            f_i = lambda r: -r*p.con[i](p.con[i].x_next) + p.r[i].gamma@(r-y_i)+rho/2*np.linalg.norm(r-y_i)**2
            constraint = LinearConstraint(np.eye(1),0)
            r_i = minimize(f_i,np.zeros(len(p.r[i].varID)),constraints=constraint).x
            p.r[i].x_next = r_i
            #the square norm of x-y

        #above is the update of local variables

        #then update the global variables
        #derive the index together is better, but here we call the index again (to be optimized)
        for i in range(len(z)):
            z[i] = np.sum([p.obj[z_obj_id[i][k][0]].x_next[z_obj_id[i][k][1]] for k in range(len(z_obj_id[i]))])+np.sum([p.con[z_con_id[i][m][0]].x_next[z_con_id[i][m][1]] for m in range(len(z_con_id[i]))])
            #z[i] += np.sum([p.obj[z_obj_id[i][k][0]].gamma[z_obj_id[i][k][1]] for k in range(len(z_obj_id[i]))])/rho+np.sum([p.con[z_con_id[i][m][0]].gamma[z_con_id[i][m][1]] for m in range(len(z_con_id[i]))])/rho
            z[i] = z[i]/(len(z_obj_id[i])+len(z_con_id[i]))

        for i in range(len(y)):
            y[i] = p.r[i].x_next[0] + 1/rho*p.r[i].gamma[0]
            y[i] = np.maximum(0,y[i])

        for i in range(num_o):
            p.obj[i].gamma = p.obj[i].gamma + rho*(p.obj[i].x_next-z[p.obj[i].varID])
        for i in range(2*num_o+1):
            p.con[i].gamma = p.con[i].gamma + rho*(p.con[i].x_next-z[p.con[i].varID])
        for i in range(num_o*2+1):
            p.r[i].gamma = p.r[i].gamma + rho*(p.r[i].x_next-y[p.r[i].varID])
        output = sum([p.obj[k](p.obj[k].x_next) for k in range(5)])
        print("step: "+str(t)+", obj: "+str(output))
    print("z: "+str(z))


    pass
    #compare with fmincon
    f = lambda x: sum([p.obj[k](x[p.obj[k].varID]) for k in range(num_o)])   

    x0 = np.random.randn(num_o)
    A = np.row_stack((np.eye(num_o),np.ones((1,num_o))))
    constraint = LinearConstraint(A,0,u_b)
    res = minimize(f,x0,constraints=constraint)
    print("centralized opt: ")
    print(res)

    
