"""
Base SegFormer Encoder with BiFPN
"""

import torch
import torch.nn.functional as F
from torch import nn
from mmcv.cnn import ConvModule

import FFESNet.models.mit as mit
# from pyramids.bifpn import 
from FFESNet.models.pyramids.pan import PAN

class FFESNet(nn.Module):
    """
    FFESNet with BiFPN
    """
    def __init__(self, model_type = 'B0', dropout = 0.1, embedding_dim = 160):
        """
        Init function
        """
        super(FFESNet, self).__init__()
        self.model_type = model_type

        # ----------------------------------------------------------------
        # Feature extractor
        # ----------------------------------------------------------------
        if self.model_type == 'B0':
            self.backbone = mit.mit_b0()
        if self.model_type == 'B1':
            self.backbone = mit.mit_b1()
        if self.model_type == 'B2':
            self.backbone = mit.mit_b2()
        if self.model_type == 'B3':
            self.backbone = mit.mit_b3()
        if self.model_type == 'B4':
            self.backbone = mit.mit_b4()
        if self.model_type == 'B5':
            self.backbone = mit.mit_b5()
        
        self._init_weights()  # load pretrain
        

        # ----------------------------------------------------------------
        # Pyramid
        # ----------------------------------------------------------------
        self.pyramid =  PAN(
                            num_levels=4,
                            in_channels=[
                                            self.backbone.embed_dims[0],
                                            self.backbone.embed_dims[1],
                                            self.backbone.embed_dims[2],
                                            self.backbone.embed_dims[3]
                                        ],
                            out_channels=64,
                        )

        # ----------------------------------------------------------------
        # Prediction
        # ----------------------------------------------------------------
        self.linear_pred = nn.Sequential(
            nn.Dropout(p = dropout),
            nn.Conv2d(256, 1, kernel_size=1),
        )
        

    def _init_weights(self):
        """
        Init pretrained
        """
        if self.model_type == 'B0':
            pretrained_dict = torch.load('/home/ptn/Storage/FFESNet/FFESNet/models/pretrained/mit_b0.pth')
        if self.model_type == 'B1':
            pretrained_dict = torch.load('/home/ptn/Storage/FFESNet/FFESNet/models/pretrained/mit_b1.pth')
        if self.model_type == 'B2':
            pretrained_dict = torch.load('/home/ptn/Storage/FFESNet/FFESNet/models/pretrained/mit_b2.pth')
        if self.model_type == 'B3':
            pretrained_dict = torch.load('/home/ptn/Storage/FFESNet/FFESNet/models/pretrained/mit_b3.pth')
        if self.model_type == 'B4':
            pretrained_dict = torch.load('/home/ptn/Storage/FFESNet/FFESNet/models/pretrained/mit_b4.pth')
        if self.model_type == 'B5':
            pretrained_dict = torch.load('/home/ptn/Storage/FFESNet/FFESNet/models/pretrained/mit_b5.pth')
            
        model_dict = self.backbone.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.backbone.load_state_dict(model_dict)
        print("Successfully loaded pretrained!!!!")
        
        
    def forward(self, x):
        # ----------------------------------------------------------------
        # Feature extractor
        # ----------------------------------------------------------------
        B = x.shape[0]
        
        # Stage 1
        out_1, H, W = self.backbone.patch_embed1(x)
        for i, blk in enumerate(self.backbone.block1):
            out_1 = blk(out_1, H, W)
        out_1 = self.backbone.norm1(out_1)
        out_1 = out_1.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  #(Batch_Size, self.backbone.embed_dims[0], 88, 88)
        
        # Stage 2
        out_2, H, W = self.backbone.patch_embed2(out_1)
        for i, blk in enumerate(self.backbone.block2):
            out_2 = blk(out_2, H, W)
        out_2 = self.backbone.norm2(out_2)
        out_2 = out_2.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  #(Batch_Size, self.backbone.embed_dims[1], 44, 44)
        
        # Stage 3
        out_3, H, W = self.backbone.patch_embed3(out_2)
        for i, blk in enumerate(self.backbone.block3):
            out_3 = blk(out_3, H, W)
        out_3 = self.backbone.norm3(out_3)
        out_3 = out_3.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  #(Batch_Size, self.backbone.embed_dims[2], 22, 22)
        
        # Stage 4
        out_4, H, W = self.backbone.patch_embed4(out_3)
        for i, blk in enumerate(self.backbone.block4):
            out_4 = blk(out_4, H, W)
        out_4 = self.backbone.norm4(out_4)
        out_4 = out_4.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  #(Batch_Size, self.backbone.embed_dims[3], 11, 11)

        # ----------------------------------------------------------------
        # Pyramid and prediction
        # ----------------------------------------------------------------
        p1, p2, p3, p4 = self.pyramid([out_1, out_2, out_3, out_4])
        
        p4 = F.interpolate(p4, scale_factor=8, mode='bilinear', align_corners=False)
        p3 = F.interpolate(p3, scale_factor=4, mode='bilinear', align_corners=False)
        p2 = F.interpolate(p2, scale_factor=2, mode='bilinear', align_corners=False)

        out = self.linear_pred(torch.cat([p1, p2, p3, p4], dim=1))
        
        return out
    

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = FFESNet('B0')
    model.to(device)

    # --------------------------------------------------------------------
    # Feedforward
    # --------------------------------------------------------------------
    x = torch.rand(1,3,384,384).to(device)
    y_pred = model(x)
    print('Successfully feedfoward!!!')

    # --------------------------------------------------------------------
    # Backprop
    # --------------------------------------------------------------------
    y = torch.rand(1,1,384,384).to(device)
    y_pred = F.upsample(y_pred, size=y.shape[2:], mode='bilinear', align_corners=False)
    y_pred = y_pred.sigmoid()
    loss = F.binary_cross_entropy_with_logits(y_pred, y, reduction='mean')

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print('Successfully backprop!!!')


if __name__=="__main__":
    main()