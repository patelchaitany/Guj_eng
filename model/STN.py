# first we define the control points
# fx(x,y) = a_1 + a_x * x + a_y *y + sum(w_ix * U(r_i))
# set of the constrain are the sum(w_i) = 0
# set of the constrain are the sum(w_i * x_i) = 0
# set of the constrain are the sum(w_i * y_i) = 0

# then we need to define the targate position of the control points
# then we need to calculate the U which is the r^2log(r) where r is the r = sqrt((x-x0)^2 + (y-y0)^2) where x0 and y0 are the control points
# then we need to define the P matrix which is control points
# then we need to define the weight matrix for the x and y
# final form which looks like this
# suppose i have N control points
# then P -> [Nx3]
# then P.T -> [3xN]
# then U -> [NxN] which is sumetric matrix
# [U P] -> [Nx(N+3)]
# [P.T 0] -> [3x(N+3)]
# [
# [U P]
# [P.T 0]
# ] -> [(N+3)x(N+3)] now suppose this whole matrix is A
#
# the constrains are written like this A*W = [z,0]
# W -> [(N+3)x1] =  [W,a]
# UW = [z,0]

# for the new points we need to calculate this formula
# fx(x,y) = a_1 + a_x * x + a_y *y + sum(w_ix * U(r_i))



# %% cell1

import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F



class STN(nn.Module):

    def __init__(self,num_control_points,I_r_size,I_num_channels = 1):
        super(STN,self).__init__()
        self.I_r_size = I_r_size
        self.I_num_channels = I_num_channels
        self.num_control_points = num_control_points

        self.localization_net = LocalisationNet(num_control_points,I_num_channels)
        self.grid_generator = GridGenrator(num_control_points,I_r_size)

    def forward(self,x):

        batch_C_prime = self.localization_net(x) # dimenstion are [B,control_points,2]
        batch_P_prime = self.grid_generator.build_P_prime(batch_C_prime) # dimenstions are [B,(Nx)x(Ny),2]
        batch_P_prime = batch_P_prime.view(-1,self.I_r_size[0],self.I_r_size[1],2)

        if torch.__version__ > "1.2.0":
            transformed_x = F.grid_sample(x, batch_P_prime, padding_mode='border', align_corners=True) # internally bilinear interpolation is used
        else:
            transformed_x = F.grid_sample(x,batch_P_prime, padding_mode='border')

        return transformed_x

class LocalisationNet(nn.Module):

    def __init__(self,num_control_points,I_num_channels):
        super(LocalisationNet,self).__init__()
        self.num_control_points = num_control_points
        self.I_num_channels = I_num_channels
        self.cov = nn.Sequential(
            nn.Conv2d(in_channels=self.I_num_channels, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False), # After -> [B,64,H,W]
            nn.BatchNorm2d(64), nn.ReLU(True), # After -> [B,64,H,W]
            nn.MaxPool2d(2, 2), # After -> [B,64,H/2,W/2]
            nn.Conv2d(64, 128, 3, 1, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True), # After -> [B,128,H/2,W/2]
            nn.MaxPool2d(2, 2), # After -> [B,128,H/4,W/4]
            nn.Conv2d(128, 256, 3, 1, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(True), # After -> [B,256,H/4,W/4]
            nn.MaxPool2d(2, 2), # After -> [B,256,H/8,W/8]
            nn.Conv2d(256, 512, 3, 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(True), # After -> [B,512,H/8,W/8]
            nn.AdaptiveAvgPool2d(1) # After -> [B,512,1,1]
        )

        self.localization_fc1 = nn.Sequential(nn.Linear(512,256),nn.ReLU(True))
        self.localization_fc2 = nn.Linear(256,2*self.num_control_points)

        self.localization_fc2.weight.data.fill_(0)

        ctrl_pts_x = np.linspace(-1.0,1.0,int(self.num_control_points/2))
        ctrl_pts_y_top = np.ones(int(self.num_control_points/2))*-1.0
        ctrl_pts_y_bottom = np.ones(int(self.num_control_points/2))*1.0
        ctrl_pts_bottom = np.stack([ctrl_pts_x,ctrl_pts_y_bottom],axis = 1)
        ctrl_pts_top = np.stack([ctrl_pts_x,ctrl_pts_y_top],axis = 1)
        initial_bias = np.concatenate([ctrl_pts_bottom,ctrl_pts_top],axis = 0)
        self.localization_fc2.bias.data = torch.tensor(initial_bias).float().view(-1)

    def forward(self,x):
        batch_size = x.size(0)
        features = self.cov(x).view(batch_size,-1)
        batch_C_prime = self.localization_fc2(self.localization_fc1(features)).view(batch_size,self.num_control_points,2)

        return batch_C_prime

class GridGenrator(nn.Module):

    def __init__(self,num_control_points,I_r_size):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        super(GridGenrator,self).__init__()
        self.control_points = num_control_points
        self.I_r_height,self.I_r_width = I_r_size
        self.C = self._Build_C(self.control_points)
        self.P = self._build_P(self.I_r_height,self.I_r_width)
        self.register_buffer('inv_delta_C',torch.tensor(self._build_inv_delta_C(self.control_points,self.C)).float())
        self.register_buffer('P_hat',torch.tensor(self.build_P_hat(self.control_points,self.C,self.P)).float())

    def _Build_C(self,control_points):
        ctrl_pts_x = np.linspace(-1.0,1.0,int(control_points/2))
        ctrl_pts_y_top = np.ones(int(control_points/2))*-1.0
        ctrl_pts_y_bottom = np.ones(int(control_points/2))*1.0
        ctrl_pts_bottom = np.stack([ctrl_pts_x,ctrl_pts_y_bottom],axis = 1)
        ctrl_pts_top = np.stack([ctrl_pts_x,ctrl_pts_y_top],axis = 1)
        ctrl_pts = np.concatenate([ctrl_pts_bottom,ctrl_pts_top],axis = 0)

        return ctrl_pts # dimenstion are [control_points,2]

    def _build_inv_delta_C(self,control_points,C):

        delta_C = np.zeros((control_points,control_points))

        for i in range(control_points):
            for j in range(control_points):
                delta_C[i,j] = np.linalg.norm(C[i]-C[j])
                delta_C[j,i] = np.linalg.norm(C[i]-C[j])

        np.fill_diagonal(delta_C,1)
        hat_C = (delta_C**2)*np.log(delta_C**2)

        delta_C = np.concatenate(
            [
                np.concatenate([np.ones((control_points,1)),C,hat_C],axis= 1), # dimenstions are [control_points,control_points+3]
                np.concatenate([np.zeros((2,3)),np.transpose(C)],axis = 1), # dimenstions are [2,control_points+3]
                np.concatenate([np.zeros((1,3)),np.ones((1, control_points))],axis=1) # dimenstions are [1,control_points+3]
            ],
            axis = 0
        ) # dimenstions are [control_points+3,control_points+3]

        inv_delta_C = np.linalg.inv(delta_C)

        return inv_delta_C  # dimenstions are [control_points+3,control_points+3]

    def _build_P(self,I_r_height,I_r_width):

        # here we are creating a meshgrid of the image
        # the meshgrid is in such a way that it is normalized
        # why we are adding one here becuse we don't want edge points becuse our initial control points are the edge points

        I_r_x = (np.arange(-I_r_width, I_r_width, 2)+ 1)/I_r_width
        I_r_y = (np.arange(-I_r_height,I_r_height,2)+ 1)/I_r_height
        P = np.stack(np.meshgrid(I_r_x,I_r_y),axis = 2).reshape(-1,2)

        return P # dimenstion are [(Nx)x(Ny),2]

    def build_P_hat(self,control_points,C,P):

        # P_hat is calculating the given function in parellel manner
        # fx(x,y) = a_1 + a_x * x + a_y *y + sum(w_ix * U(r_i))

        n = P.shape[0] # dimenstion are [(Nx)x(Ny),2] || # number of point in this manner -> ([X,Y1],[X,Y2]....[X,Yn]) , ([X1,Y1] ,[X1,Y2]....[X1,Yn])
        P_tile = np.tile(np.expand_dims(P,axis = 1),(1,control_points,1)) # dimenstion are [(Nx)x(Ny),1,2] -> then transformation [(Nx)x(Ny),control_points,2]
        C_tile = np.expand_dims(C,axis = 0) # dimenstion are [1,control_points,2]
        P_diff = P_tile - C_tile # dimenstion are [(Nx)x(Ny),control_points,2]
        rbf_norm = np.linalg.norm(P_diff,axis = 2,keepdims = False,ord = 2) # dimenstion are [(Nx)x(Ny),control_points]
        rbf = rbf_norm**2 * np.log(rbf_norm) # dimenstion are [(Nx)x(Ny),control_points]
        P_hat = np.concatenate([np.ones((n, 1)),P,rbf],axis = 1) # dimenstion are [(Nx)x(Ny),control_points+3]

        return P_hat

    def build_P_prime(self,batch_C_prime):
        batch_size = batch_C_prime.shape[0] # dimenstion are [B,control_points,2]
        batch_inv_delta_C = self.inv_delta_C.repeat(batch_size,1,1) # dimenstion are [B,control_points+3,control_points+3]
        batch_P_hat = self.P_hat.repeat(batch_size,1,1) # dimenstion are [B,(Nx)x(Ny),control_points+2]
        batch_C_prime_with_zeros = torch.cat([batch_C_prime,torch.zeros(batch_size,3,2).to(self.device)],dim = 1) # dimenstion are [B,control_points+3,2]
        batch_T = torch.bmm(batch_inv_delta_C,batch_C_prime_with_zeros) # dimenstion are [B,control_points+3,2]
        batch_P_prime = torch.bmm(batch_P_hat,batch_T) # dimenstion are [B,(Nx)x(Ny),2]

        return batch_P_prime # dimenstion are [B,(Nx)x(Ny),2]

# %% cell 2

if __name__ == '__main__':
    num_control_points = 16
    I_size = (28,28)
    I_r_size = (60,60)
    I_num_channels = 3
    stn = STN(num_control_points,I_r_size,I_num_channels)
    x = torch.randn(10,3,28,28)
    y = stn(x)
    print(y.shape)
