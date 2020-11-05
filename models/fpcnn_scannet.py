import torch
import torch.nn as nn
from fpconv.pointnet2.pointnet2_modules import PointnetFPModule
import fpconv.pointnet2.pytorch_utils as pt_utils
from fpconv.base import AssemRes_BaseBlock
from fpconv.fpconv import FPConv4x4_BaseBlock, FPConv6x6_BaseBlock


NPOINT = 8192
NPOINTS = [NPOINT // 2, NPOINT // 8,  NPOINT // 32,  NPOINT // 128]
RADIUS = [0.1, 0.2, 0.4, 0.8, 1.6]
NSAMPLE = [32, 32, 32, 32, 16]
MLPS = [[32,32], [64,64], [128,128], [256,256], [512,512]]
FP_MLPS = [[64,64], [128,64], [256,128], [512,256]]
CLS_FC = [64]
DP_RATIO = 0.5


def get_model(num_class, input_channels=3, num_pts=None):
    """
    Construct a model.

    Args:
        num_class: (int): write your description
        input_channels: (todo): write your description
        num_pts: (int): write your description
    """
    if num_pts is None:
        num_pts = NPOINT
    return FPCNN_ScanNet(num_pts, num_class, input_channels)


class FPCNN_ScanNet(nn.Module):
    def __init__(self, num_pts, num_class, input_channels, use_xyz=False):
        """
        Initialize the channel.

        Args:
            self: (todo): write your description
            num_pts: (int): write your description
            num_class: (int): write your description
            input_channels: (todo): write your description
            use_xyz: (bool): write your description
        """
        # input_channels: input feature channels (not include xyz)
        super().__init__()
        NPOINT = num_pts
        NPOINTS = [NPOINT // 2, NPOINT // 8,  NPOINT // 32,  NPOINT // 128]
        print(NPOINTS)

        self.SA_modules = nn.ModuleList()
        self.conv0 = AssemRes_BaseBlock(
                        CONV_BASE=FPConv6x6_BaseBlock,
                        npoint=None,
                        radius=RADIUS[0],
                        nsample=NSAMPLE[0],
                        channel_list=[input_channels] + MLPS[0],
                        use_xyz=use_xyz)
        
        channel_in = MLPS[0][-1]
        skip_channel_list = [channel_in]
        for k in range(NPOINTS.__len__()):
            mlps = [MLPS[k+1].copy()]
            channel_out = 0

            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]

            print(mlps[0], RADIUS[k], RADIUS[k+1])
            if k < 2:
                self.SA_modules.append(
                    AssemRes_BaseBlock(
                        CONV_BASE=FPConv6x6_BaseBlock,
                        npoint=NPOINTS[k],
                        nsample=NSAMPLE[k],
                        radius=RADIUS[k],
                        channel_list=mlps[0],
                        nsample_ds=NSAMPLE[k+1],
                        radius_ds=RADIUS[k+1],
                        use_xyz=use_xyz))
            else:
                self.SA_modules.append(
                    AssemRes_BaseBlock(
                        CONV_BASE=FPConv4x4_BaseBlock,
                        npoint=NPOINTS[k],
                        nsample=NSAMPLE[k],
                        radius=RADIUS[k],
                        channel_list=mlps[0],
                        nsample_ds=NSAMPLE[k+1],
                        radius_ds=RADIUS[k+1],
                        use_xyz=use_xyz))
            skip_channel_list.append(channel_out)
            channel_in = channel_out

        self.FP_modules = nn.ModuleList()

        for k in range(FP_MLPS.__len__()):
            pre_channel = FP_MLPS[k + 1][-1] if k + 1 < len(FP_MLPS) else channel_out
            mlp = [pre_channel + skip_channel_list[k]] + FP_MLPS[k]
            print(mlp)
            self.FP_modules.append(PointnetFPModule(mlp=mlp))

        cls_layers = []
        pre_channel = FP_MLPS[0][-1]
        for k in range(0, CLS_FC.__len__()):
            cls_layers.append(pt_utils.Conv2d(pre_channel, CLS_FC[k], bn=True))
            pre_channel = CLS_FC[k]
        cls_layers.append(pt_utils.Conv2d(pre_channel, num_class, activation=None, bn=False))
        cls_layers.insert(1, nn.Dropout(0.5))
        self.cls_layer = nn.Sequential(*cls_layers)

    def _break_up_pc(self, pc):
        """
        Return the features into a features.

        Args:
            self: (todo): write your description
            pc: (todo): write your description
        """
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None)

        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor):
        """
        Forward computation

        Args:
            self: (todo): write your description
            pointcloud: (todo): write your description
            torch: (todo): write your description
            cuda: (todo): write your description
            FloatTensor: (todo): write your description
        """
        xyz, features = self._break_up_pc(pointcloud)

        _, features = self.conv0(xyz, features)
        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i])

        fn_feats = l_features[0].unsqueeze(-1) # B, C, N, 1
        pred_cls = self.cls_layer(fn_feats).squeeze(-1).transpose(1, 2).contiguous() # B, N, C
        return pred_cls
        