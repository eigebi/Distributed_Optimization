class data:
        def __init__(self,size=5000):
            self.lambda_past = []
            self.x_past = []
            self.L_past = []
            self.size = size
        def reset(self):
            self.lambda_past = []
            self.x_past = []
            self.L_past = []
        def append(self, x, r, L):
            self.lambda_past.append(r.detach().numpy())
            self.x_past.append(x.detach().numpy())
            self.L_past.append(L)
            if len(self.lambda_past)>self.size:
                self.lambda_past.pop(0)
                self.x_past.pop(0)
                self.L_past.pop(0)