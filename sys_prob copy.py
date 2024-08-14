import numpy as np
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
import scipy
np.random.seed(1008)

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


def exam_func(arg):
    # we return a polynomial function give the arg as coefficients
    # the function has 3 variables with maximum degree 2
    
    return lambda x: x @ arg[:-2] @ x + arg[-1]


    




class prob:
    def __init__(self):
        self.obj = []
        self.con = []
        self.r = []


if __name__ == '__main__':
    f_s = []
    non_convex = True
    jac = []
    for _ in range(5):
        temp = np.random.randn(3,3)
        temp = temp @ temp.T
        temp2 = np.random.randn(3)*15

        f_s.append(lambda x: x @ temp @ x + temp2 @ x)
        jac.append(lambda x: 2*temp @ x + temp2)
        #generate 3 integers from (1,5) without repetition
        #np.random.choice(5,3,replace=False)
    p = prob()
    for f in f_s:
        var_select = np.random.choice(5,3,replace=False)
        p.obj.append(BaseProblem(f,var_select))
    u_b = np.zeros(6)
    for i in range(5): 
        u_bound = np.random.randint(1,16)
        p.con.append(BaseProblem(lambda x: x - u_bound ,[i]))
        u_b[i]=u_bound
    for i in range(5):
        p.con.append(BaseProblem(lambda x: -x ,[i]))
    p.con.append(BaseProblem(lambda x: np.sum(x)-30 ,np.arange(5)))
    u_b[-1] = 30
    for i in range(11):
        p.r.append(BaseProblem(lambda x: x , [i],is_min=False))
    # global x
    z = np.random.randn(5)
    # global lambda
    y = np.abs(np.random.randn(11)).reshape(-1,1)
    #gamma = np.abs(np.random.randn(11)) this gamma is local
    for i in range(11):
        p.r[i].x_next =y[i]


    T = 500
    rho = 2


    # need an index table to connect z to all the local variables, which is inverted to the varID
    # can be wrapped into a function
    z_obj_id = []
    z_con_id = []
    for i in range(5):
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
        for i in range(5):
            # to be wrapped into optimize()
            z_i= z[p.obj[i].varID]
            if non_convex:
                f_i = lambda x: jac[i](x)@(x-z_i)+p.obj[i].gamma@(x-z_i)+rho/2*np.linalg.norm(x-z_i)**2
            else:
                f_i = lambda x: p.obj[i](x)+p.obj[i].gamma@(x-z_i)+rho/2*np.linalg.norm(x-z_i)**2
            x_i = minimize(f_i,np.zeros(len(p.obj[i].varID))).x
            p.obj[i].x_next = x_i
            #the square norm of x-z
        for i in range(11):
            # to be wrapped into optimize()
            z_i= z[p.con[i].varID]
            #f_i = lambda x: p.r[i].x_next*p.con[i].func(x)+p.con[i].gamma@(x-z_i)+rho/2*np.linalg.norm(x-z_i)**2
            f_i = lambda x: p.r[i].x_next*p.con[i](x)+p.con[i].gamma@(x-z_i)+rho/2*np.linalg.norm(x-z_i)**2
            x_i = minimize(f_i,np.zeros(len(p.con[i].varID))).x
            p.con[i].x_next = x_i
            #the square norm of x-y
            #it is better to store the con_func value to serve the next step update

        #update lambda in the lagrangian reformulation
        for i in range(11):
            # to be wrapped into optimize()
            y_i= y[p.r[i].varID]
            f_i = lambda r: -r*p.con[i](p.con[i].x_next) + p.r[i].gamma@(r-y_i)+rho/2*np.linalg.norm(r-y_i)**2
            constraint = LinearConstraint(np.eye(1),0)
            r_i = minimize(f_i,np.zeros(len(p.r[i].varID)),constraints=constraint).x
            
            p.r[i].x_next += 0.001* (p.con[i](z[p.con[i].varID]))
            p.r[i].x_next = np.zeros((1,1)) if p.r[i].x_next<0 else p.r[i].x_next

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

        for i in range(5):
            p.obj[i].gamma = p.obj[i].gamma + rho*(p.obj[i].x_next-z[p.obj[i].varID])
        for i in range(11):
            p.con[i].gamma = p.con[i].gamma + rho*(p.con[i].x_next-z[p.con[i].varID])
        for i in range(11):
            p.r[i].gamma = p.r[i].gamma + rho*(p.r[i].x_next-y[p.r[i].varID])
        output = sum([p.obj[k](p.obj[k].x_next) for k in range(5)])
        print("step: "+str(t)+", obj: "+str(output))
    print("z: "+str(z))


    pass
    #compare with fmincon
    f = lambda x: sum([p.obj[k](x[p.obj[k].varID]) for k in range(5)])   

    x0 = np.random.randn(5)
    A = np.row_stack((np.eye(5),np.ones((1,5))))
    constraint = LinearConstraint(A,0,u_b)
    res = minimize(f,x0,constraints=constraint)
    print("centralized opt: ")
    print(res)

    