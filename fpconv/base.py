import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from fpconv.pointnet2 import pointnet2_utils
from fpconv.pointnet2 import pytorch_utils as pt_utils
relu_alpha = 0.2


class PointNet(nn.Module):
    def __init__(self, mlp, pool='max', bn=True):
        super().__init__()
        self.mlp = pt_utils.SharedMLP(mlp, bn=bn, activation=nn.LeakyReLU(negative_slope=relu_alpha, inplace=True))

    def forward(self, pcd):
        '''
        :param pcd: B, C, npoint, nsample
        :return:
            new_pcd: B, C_new, npoint, 1
        '''
        new_pcd = self.mlp(pcd) # B, C_new, npoint, nsample
        new_pcd = F.max_pool2d(new_pcd, kernel_size=[1, new_pcd.size(3)]) # B, C_new, npoint, 1
        return new_pcd


class ProjWeightModule(nn.Module):
    def __init__(self, mlp_pn, mlp_wts, map_size, bn=True):
        super().__init__()
        map_len = map_size ** 2
        mlp_pn = [3] + mlp_pn
        mlp_wts = [mlp_pn[-1] + 3] + mlp_wts + [map_len] # 3+C_new => map_len
        self.pn_layer = PointNet(mlp_pn, bn=bn)
        self.wts_layer = pt_utils.SharedMLP(mlp_wts, 
                                            bn=bn, 
                                            activation=nn.LeakyReLU(negative_slope=relu_alpha, inplace=True))

    def forward(self, xyz):
        '''
        :param xyz: B, 3, npoint, nsample <local>
        :return:
            proj_wts: B, map_len, npoint, nsample
        '''
        nsample = xyz.size(3)
        dist_feat = self.pn_layer(xyz) # B, C_new, npoint, 1
        dist_feat = dist_feat.expand(-1, -1, -1, nsample) # B, C_new, npoint, nsample
        dist_feat = torch.cat([xyz, dist_feat], dim=1) # B, C_new+3, npoint, nsample

        proj_wts = self.wts_layer(dist_feat) # B, map_len, npoint, nsample
        return proj_wts


class PN_Block(nn.Module):
    def __init__(self, in_channel, out_channel, bn=True, activation=True):
        # Shared MLPs
        super().__init__()
        self.conv = pt_utils.Conv2d(in_size=in_channel, 
                                    out_size=out_channel, 
                                    kernel_size=(1,1), 
                                    bn=bn, 
                                    activation=nn.LeakyReLU(negative_slope=relu_alpha, inplace=True) if activation else None)

    def forward(self, pcd):
        '''
        :param pcd: B, C_in, npoint
        :return:
            new_pcd: B, C_out, npoint
        '''
        pcd = pcd.unsqueeze(-1)
        return self.conv(pcd).squeeze(-1)


class Pooling_Block(nn.Module):
    def __init__(self, radius, nsample, in_channel, out_channel, npoint=None, bn=True, activation=True):
        super().__init__()
        self.radius = radius
        self.nsample = nsample
        self.npoint = npoint
        self.conv = PN_Block(in_channel, out_channel, bn=bn, activation=activation)

    def forward(self, xyz, feats, new_xyz=None):
        '''
        :param pcd: B, C_in, N
        :return:
            new_pcd: B, C_out, np
        '''
        if new_xyz is None:
            assert self.npoint is not None
            xyz_flipped = xyz.transpose(1, 2).contiguous() # B,3,npoint
            idx = pointnet2_utils.furthest_point_sample(xyz, self.npoint) # B,npoint
            new_xyz_flipped = pointnet2_utils.gather_operation(xyz_flipped, idx) # B,3,npoint
            new_xyz = new_xyz_flipped.transpose(1, 2).contiguous() # B,npoint,3

        idx = pointnet2_utils.ball_query(self.radius, self.nsample, xyz, new_xyz)
        gped_feats = pointnet2_utils.grouping_operation(feats, idx) # B,C,np,ns
        gped_feats = F.max_pool2d(gped_feats, kernel_size=[1, self.nsample]) # B,C,np,1
        gped_feats = gped_feats.squeeze(-1) # B,C,np

        return self.conv(gped_feats)


class Resnet_BaseBlock(nn.Module):
    def __init__(self, FPCONV,
            npoint, nsample, radius, in_channel, out_channel, bn=True, use_xyz=False):
        '''
        pcd => 1x1 conv <relu+bn> => tconv <relu+bn> => 1x1 conv <bn>
        shortcut: pcd => (max_pooling) => 1x1 conv <bn> [apply projection shortcut]
        :param npoint: set to None to ignore 'max_pooling'
        :param nsample, radius: params related to grouper
        '''
        super().__init__()
        self.keep_pcd = npoint is None
        self.is_im = in_channel == out_channel
        self.mid_channel = out_channel // 2 # <Bottleneck Design Block>

        self.conv1 = PN_Block(in_channel=in_channel,
                              out_channel=self.mid_channel,
                              bn=bn)
                                
        self.conv2 = FPCONV(npoint=npoint, 
                            nsample=nsample,
                            radius=radius,
                            in_channel=self.mid_channel,
                            out_channel=self.mid_channel,
                            bn=bn,
                            use_xyz=use_xyz)
        
        self.conv3 = PN_Block(in_channel=self.mid_channel,
                              out_channel=out_channel,
                              bn=bn,
                              activation=False)

        if self.keep_pcd and not self.is_im:
            self.sonv0 = PN_Block(in_channel=in_channel,
                                  out_channel=out_channel,
                                  bn=bn,
                                  activation=False)
        elif not self.keep_pcd:
            self.sonv0 = Pooling_Block(radius=radius,
                                       nsample=nsample,
                                       in_channel=in_channel,
                                       out_channel=out_channel,
                                       bn=bn,
                                       activation=False)

    def forward(self, xyz, feats, new_xyz=None):
        assert (self.keep_pcd and new_xyz is None) or not self.keep_pcd, 'invalid new_xyz.'
        
        new_feats = self.conv1(feats)
        new_xyz, new_feats = self.conv2(xyz, new_feats, new_xyz)
        new_feats = self.conv3(new_feats)
        shc_feats = feats

        if self.keep_pcd and not self.is_im: # if in != out, applt an additional projection mlp
            shc_feats = self.sonv0(shc_feats) # mlp
        if not self.keep_pcd: # not keep pcd, apply fpconv with fps
            shc_feats = self.sonv0(xyz, feats, new_xyz) # pooling + mlp 

        new_feats = F.leaky_relu(shc_feats + new_feats, negative_slope=relu_alpha ,inplace=True)
        return new_xyz, new_feats


class AssemRes_BaseBlock(nn.Module):
    def __init__(self, CONV_BASE,
            npoint, nsample, radius, channel_list, nsample_ds=None, radius_ds=None, bn=True, use_xyz=False):
        '''
        Apply downsample and conv on input pcd
        :param npoint: the number of points to sample
        :param nsample: the number of neighbors to group when conv
        :param radius: radius of ball query to group neighbors
        :param channel_list: List<a, c, c, ...>, the elements from <1> to the last must be the same
        '''
        super().__init__()
        if nsample_ds is None:
            nsample_ds = nsample
        if radius_ds is None:
            radius_ds = radius

        self.conv_blocks = nn.ModuleList()
        for i in range(len(channel_list) - 1):
            in_channel = channel_list[i]
            out_channel = channel_list[i+1]
            self.conv_blocks.append(Resnet_BaseBlock(FPCONV=CONV_BASE,
                                                     npoint=npoint if i == 0 else None,
                                                     nsample=nsample if i == 0 else nsample_ds,
                                                     radius=radius if i == 0 else radius_ds,
                                                     in_channel=in_channel,
                                                     out_channel=out_channel,
                                                     bn=bn,
                                                     use_xyz=use_xyz))

    def forward(self, xyz, feats, new_xyz=None):
        for i, block in enumerate(self.conv_blocks):
            xyz, feats = block(xyz, feats, new_xyz)
        
        return xyz, feats
