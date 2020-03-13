import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from fpconv.pointnet2 import pointnet2_utils
from fpconv.pointnet2 import pytorch_utils as pt_utils
from fpconv import base
relu_alpha = 0.2


class FPConv4x4_BaseBlock(nn.Module):
    def __init__(self, npoint, nsample, radius, in_channel, out_channel, bn=True, use_xyz=False):
        
        super().__init__()
        print('fpconv4x4 init:', npoint, nsample, radius, in_channel, out_channel)
        self.npoint = npoint
        self.nsample = nsample
        self.keep_pcd = npoint is None
        self.use_xyz = use_xyz

        self.grouper = pointnet2_utils.QueryAndGroupLocal(radius, nsample)
        self.wts_layer = base.ProjWeightModule(mlp_pn=[8,16], mlp_wts=[16], map_size=4, bn=bn)
        if use_xyz:
            in_channel += 3

        self.proj_conv = pt_utils.Conv2d(in_size=in_channel, 
                                         out_size=out_channel, 
                                         kernel_size=(16,1), 
                                         bn=bn,
                                         activation=nn.LeakyReLU(negative_slope=relu_alpha, inplace=True))

    def forward(self, xyz, features, new_xyz=None):
        '''
        :param xyz: B,N,3
        :param features: B,C,N
        :returns:
            new_xyz: B,np,3
            new_feats: B,C,np
        '''
        # sample new xyz
        if not self.keep_pcd and new_xyz is None:
            xyz_flipped = xyz.transpose(1, 2).contiguous() # B,3,npoint
            idx = pointnet2_utils.furthest_point_sample(xyz, self.npoint) # B,npoint
            new_xyz_flipped = pointnet2_utils.gather_operation(xyz_flipped, idx) # B,3,npoint
            new_xyz = new_xyz_flipped.transpose(1, 2).contiguous() # B,npoint,3
        elif new_xyz is not None:
            self.npoint = new_xyz.size(1)
        else: # keep pcd
            new_xyz = xyz
            self.npoint = new_xyz.size(1)
        
        # get distribution vector
        grouped_xyz, grouped_feats = self.grouper(xyz, new_xyz, features)
        proj_wts = self.wts_layer(grouped_xyz) # B,ml+1,np,ns
        if self.use_xyz:
            grouped_feats = torch.cat([grouped_xyz, grouped_feats], dim=1)

        # normalize weights
        # normalize at dim 1 <ml>
        proj_wts2_ = proj_wts ** 2 # B, ml, np, ns
        proj_wts_sum = torch.sum(proj_wts2_, dim=1, keepdim=True) # B, 1, np, ns
        proj_wts_sum = torch.max(proj_wts_sum, torch.tensor(1e-8).cuda())
        proj_wts_sum = torch.sqrt(proj_wts_sum) # B, 1, np, ns
        proj_wts = proj_wts / proj_wts_sum

        # normalize at dim 3 <nsample>
        proj_wts_sum = torch.sum(proj_wts2_, dim=3, keepdim=True) # B,ml,np,1
        proj_wts_sum = torch.max(proj_wts_sum, torch.tensor(1e-8).cuda())
        proj_wts_sum = torch.sqrt(proj_wts_sum) # B, 1, np, ns
        proj_wts_sum = torch.max(proj_wts_sum, torch.tensor(1.0).cuda())
        proj_wts = proj_wts / proj_wts_sum # B,ml,np,ns

        # projection
        proj_wts = proj_wts.transpose(1,2) # B, np, ml, ns
        grouped_feats = grouped_feats.permute(0, 2, 3, 1) # B, C, np, bs => B, np, ns, C
        multi = proj_wts.matmul(grouped_feats)
        proj_feats = F.leaky_relu(proj_wts.matmul(grouped_feats), negative_slope=relu_alpha, inplace=True) # B, np, ml, C
        proj_feats = proj_feats.transpose(1,3) # B, C, ml, np

        # convolution
        proj_feats = self.proj_conv(proj_feats) # B, C_new, 1, np
        proj_feats = proj_feats.squeeze(2) # B, C_new, np
        
        return new_xyz, proj_feats


class FPConv6x6_BaseBlock(nn.Module):
    def __init__(self, npoint, nsample, radius, in_channel, out_channel, bn=True, use_xyz=False):

        super().__init__()
        print('fpconv6x6 init:', npoint, nsample, radius, in_channel, out_channel)
        self.npoint = npoint
        self.map_size = 6
        self.map_len = self.map_size ** 2
        self.nsample = nsample
        self.keep_pcd = npoint is None
        self.use_xyz = use_xyz

        self.grouper = pointnet2_utils.QueryAndGroupLocal(radius, nsample)
        self.wts_layer = base.ProjWeightModule(mlp_pn=[8,16,16], mlp_wts=[16,32], map_size=6, bn=bn)

        if use_xyz:
            in_channel += 3
        
        self.bias = Parameter(torch.Tensor(in_channel))
        mid_channel = in_channel
        self.proj_conv = nn.Sequential(
            pt_utils.Conv3d(in_size=in_channel, 
                            out_size=mid_channel, 
                            kernel_size=(3,3,1), 
                            bn=bn, 
                            activation=nn.LeakyReLU(negative_slope=relu_alpha, inplace=True)),
            pt_utils.Conv3d(in_size=in_channel, 
                            out_size=mid_channel, 
                            kernel_size=(3,3,1), 
                            bn=bn, 
                            activation=nn.LeakyReLU(negative_slope=relu_alpha, inplace=True)),
            pt_utils.Conv3d(in_size=mid_channel, 
                            out_size=out_channel, 
                            kernel_size=(2,2,1), 
                            bn=bn, 
                            activation=nn.LeakyReLU(negative_slope=relu_alpha, inplace=True)))
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.bias, -0.05)

    def forward(self, xyz, features, new_xyz=None):
        '''
        :param xyz: B,N,3
        :param features: B,C,N
        :returns:
            new_xyz: B,np,3
            new_feats: B,C,np
        '''

        # sample new xyz
        if not self.keep_pcd and new_xyz is None:
            xyz_flipped = xyz.transpose(1, 2).contiguous() # B,3,npoint
            idx = pointnet2_utils.furthest_point_sample(xyz, self.npoint) # B,npoint
            new_xyz_flipped = pointnet2_utils.gather_operation(xyz_flipped, idx) # B,3,npoint
            new_xyz = new_xyz_flipped.transpose(1, 2).contiguous() # B,npoint,3
        elif new_xyz is not None:
            idx = None
            self.npoint = new_xyz.size(1)
        else:
            idx = None
            new_xyz = xyz
            self.npoint = new_xyz.size(1)

        # get distribution vector
        grouped_xyz, grouped_feats = self.grouper(xyz, new_xyz, features)
        proj_wts = self.wts_layer(grouped_xyz) # B,ml,np,ns
        if self.use_xyz:
            grouped_feats = torch.cat([grouped_xyz, grouped_feats], dim=1)

        # normalize weights
        # normalize at dim 1 <ml>
        proj_wts2_ = proj_wts ** 2 # B, ml, np, ns
        proj_wts_sum = torch.sum(proj_wts2_, dim=1, keepdim=True) # B, 1, np, ns
        proj_wts_sum = torch.max(proj_wts_sum, torch.tensor(1e-8).cuda())
        proj_wts_sum = torch.sqrt(proj_wts_sum) # B, 1, np, ns
        proj_wts = proj_wts / proj_wts_sum

        # normalize at dim 3 <nsample>
        # proj_wts2_ = proj_wts ** 2 # B, ml, np, ns
        proj_wts_sum = torch.sum(proj_wts2_, dim=3, keepdim=True) # B,ml,np,1
        proj_wts_sum = torch.max(proj_wts_sum, torch.tensor(1e-8).cuda())
        proj_wts_sum = torch.sqrt(proj_wts_sum) # B, 1, np, ns
        proj_wts_sum = torch.max(proj_wts_sum, torch.tensor(1.0).cuda())
        proj_wts = proj_wts / proj_wts_sum # B,ml,np,ns

        # projection
        proj_wts = proj_wts.transpose(1,2) # B, np, ml, ns
        grouped_feats = grouped_feats.permute(0, 2, 3, 1) # B, C, np, bs => B, np, ns, C
        proj_feats = F.leaky_relu(proj_wts.matmul(grouped_feats) + self.bias, negative_slope=relu_alpha, inplace=True) # B, np, ml, C

        # reshape projection features # B, np, ml, C => B, C, ms, ms, np
        bs = proj_feats.size(0)
        proj_feats = proj_feats.transpose(1, 3) # B, C, ml, np
        proj_feats = proj_feats.view(bs, -1, self.map_size, self.map_size, self.npoint).contiguous() # B, C, ms, ms, np

        # convolution
        proj_feats = self.proj_conv(proj_feats) # B, C_new, 1, 1, np
        proj_feats = proj_feats.squeeze(3).squeeze(2) # B, C_new, np
        
        return new_xyz, proj_feats