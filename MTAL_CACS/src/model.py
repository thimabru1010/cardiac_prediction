#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 09:41:53 2021

@author: Bernhard Foellmer
"""

import torch
from torch import nn, optim

class MTALModel():
    """
    MTALModel - Multi task model
    """
    
    def __init__(self, device='cuda'):
        
        # Init params
        self.params=dict()
        self.params['lr'] = 0.005
        self.params['device'] = device

    def create(self, c_pos=4, hidden_dim=32):
        """
        Create model
        """
        class SlicePositionalEncodingEmbedding(nn.Module):
            def __init__(self, c_pos=1, max_slices=512):
                super().__init__()
                self.c_pos = c_pos
                self.embedding = nn.Embedding(max_slices, c_pos)

            def forward(self, slice_idx, spatial_size):
                """
                slice_idx: (B,) long tensor com índice do slice (0 .. max_slices-1)
                spatial_size: (H, W)
                """
                B = slice_idx.size(0)
                H, W = spatial_size

                e = self.embedding(slice_idx)      # (B, c_pos)
                e = e.view(B, self.c_pos, 1, 1)    # (B, c_pos, 1, 1)
                e = e.expand(B, self.c_pos, H, W)  # (B, c_pos, H, W)

                return e
    
        class SlicePositionalEncodingMLP(nn.Module):
            def __init__(self, c_pos=1, hidden_dim=32):
                super().__init__()
                self.c_pos = c_pos
                self.mlp = nn.Sequential(
                    nn.Linear(1, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, c_pos)
                )
            def forward(self, slice_pos, spatial_size):
                """
                slice_pos: tensor (B,) com posição contínua em [0, 1] (por ex: z_normalizado)
                spatial_size: (H, W) da imagem
                """
                B = slice_pos.size(0)
                H, W = spatial_size

                # (B,) -> (B,1)
                z = slice_pos.view(B, 1)
                e = self.mlp(z)                # (B, c_pos)

                # vira (B, c_pos, 1, 1)
                e = e.view(B, self.c_pos, 1, 1)

                # expand "repete logicamente" para (B, c_pos, H, W), sem copiar memória
                e = e.expand(B, self.c_pos, H, W)

                return e  # (B, c_pos, H, W)
            
        class Conv_down(nn.Module):
            def __init__(self, in_ch, out_ch):
                super(Conv_down, self).__init__()
                self.down = nn.Conv2d(in_ch, out_ch,  kernel_size=4, stride=2, padding=1)
                self.relu1 = nn.LeakyReLU(0.2)
                self.dropout = nn.Dropout(p=0.0)
                self.conv = nn.Conv2d(out_ch, out_ch,  kernel_size=3, stride=1, padding=1)
                self.norm = nn.BatchNorm2d(out_ch)
                self.relu2 = nn.LeakyReLU(0.2)
                self.down.weight.data.normal_(0.0, 0.1)
                self.conv.weight.data.normal_(0.0, 0.1)
        
            def forward(self, x):
                x = self.down(x)
                x = self.relu1(x)
                x = self.dropout(x)
                x = self.conv(x)
                x = self.norm(x)
                x = self.relu2(x)
                return x

        class Conv_up(nn.Module):
            def __init__(self, in_ch, out_ch, kernel_size_1=3, stride_1=1, padding_1=1, kernel_size_2=3, stride_2=1, padding_2=1):
                super(Conv_up, self).__init__()
                self.up = nn.UpsamplingBilinear2d(scale_factor=2)
                self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size_1, padding=padding_1, stride=stride_1)
                self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size_2, padding=padding_2, stride=stride_2)
                self.relu1 = nn.LeakyReLU(0.2)
                self.relu2 = nn.LeakyReLU(0.2)
                self.dropout = nn.Dropout(p=0.0)
                self.norm = nn.BatchNorm2d(out_ch)
                self.conv1.weight.data.normal_(0.0, 0.1)
                self.conv2.weight.data.normal_(0.0, 0.1)
        
            def forward(self, x1, x2):
                x1 = self.up(x1)
                x = torch.cat((x1, x2), dim=1)
                x = self.conv1(x)
                x = self.relu1(x)
                x = self.dropout(x)
                x = self.conv2(x)
                x = self.norm(x)
                x = self.relu2(x)
                return x
            
        class MTAL(nn.Module):
            def __init__(self, c_pos=0, hidden_dim=32):
                super(MTAL, self).__init__()

                #self.conv_down1 = Conv_down(props['Input_size'][2], 64)
                
                self.conv_down1 = Conv_down(16, 16)
                self.conv_down2 = Conv_down(16, 32)
                self.conv_down3 = Conv_down(32, 32)
                self.conv_down4 = Conv_down(32, 64)
                self.conv_down5 = Conv_down(64, 64)
                self.conv_down6 = Conv_down(64, 64)
                self.conv_down7 = Conv_down(64, 128)
                self.conv_down8 = Conv_down(128, 128)
                
                self.conv_up1 = Conv_up(128+128, 128)
                self.conv_up2 = Conv_up(128+64, 64)
                self.conv_up3 = Conv_up(64+64, 64)
                self.conv_up4 = Conv_up(64+64, 64)
                self.conv_up5 = Conv_up(64+32, 32)
                self.conv_up6 = Conv_up(32+32, 32)
                self.conv_up7 = Conv_up(32+16, 16)
                self.conv_up8 = Conv_up(16+16, 16)

                self.conv_up1_class = Conv_up(128+128+128, 128)
                self.conv_up2_class = Conv_up(128+64+64, 64)
                self.conv_up3_class = Conv_up(64+64+64, 64)
                self.conv_up4_class = Conv_up(64+64+64, 64)
                self.conv_up5_class  = Conv_up(64+32+32, 32)
                self.conv_up6_class  = Conv_up(32+32+32, 32)
                self.conv_up7_class  = Conv_up(32+16+16, 16)
                self.conv_up8_class  = Conv_up(16+16+16+2, 32)
                # self.conv_up6_class  = Conv_up(32+32, 32)
                # self.conv_up7_class  = Conv_up(32+16, 16)
                # self.conv_up8_class  = Conv_up(16+16+2, 32)
                
                self.conv00 = nn.Conv2d(2, 16, kernel_size=5, padding=2, stride=1)
                self.relu00 = nn.LeakyReLU(0.2)
                self.conv01 = nn.Conv2d(16, 16, kernel_size=5, padding=2, stride=1)
                self.relu01 = nn.LeakyReLU(0.2)
                
                self.conv_double_out = nn.Sequential(
                    nn.Conv2d(16, 4,  kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(4, 4,  kernel_size=3, stride=1, padding=1)
                )
                
                # self.softmax = nn.Softmax(dim=1)
                
                self.conv0 = nn.Conv2d(32, 16, kernel_size=3, padding=1, stride=1)
                self.relu0 = nn.LeakyReLU(0.2)
                self.conv1 = nn.Conv2d(16, 2, kernel_size=3, padding=1, stride=1)
                self.soft = nn.Softmax(dim=1)

                self.dropout0 = nn.Dropout(p=0.5)
                
                self.pos_encoding = SlicePositionalEncodingMLP(c_pos=c_pos, hidden_dim=hidden_dim)
                

            def forward(self, x, slice_pos=None):
                
                # if slice_pos is not None:
                #     B, C, H, W = x.shape
                #     pos_map = self.pos_encoding(slice_pos, (H, W))  # (B, c_pos, H, W)
                #     x_in = torch.cat([x, pos_map], dim=1)           # (B, 2 + c_pos, H, W)
                # else:
                x_in = x
        
                # print(x.shape)
                x00 = self.conv00(x_in)
                x00r = self.relu00(x00)
                x01 = self.conv01(x00r)
                x01r = self.relu01(x01)
                # print(x01r.shape)
                
                x1 = self.conv_down1(x01r)
                # x1_pos_map = self.pos_encoding(slice_pos, (x1.shape[2], x1.shape[3]))
                # x1 = x1 + x1_pos_map
                
                x2 = self.conv_down2(x1)
                # x2_pos_map = self.pos_encoding(slice_pos, (x2.shape[2], x2.shape[3]))
                # x2 = x2 + x2_pos_map
                
                x3 = self.conv_down3(x2)
                # x3_pos_map = self.pos_encoding(slice_pos, (x3.shape[2], x3.shape[3]))
                # x3 = x3 + x3_pos_map
                
                x4 = self.conv_down4(x3)
                # x4_pos_map = self.pos_encoding(slice_pos, (x4.shape[2], x4.shape[3]))
                # x4 = x4 + x4_pos_map
                
                x5 = self.conv_down5(x4)
                # x5_pos_map = self.pos_encoding(slice_pos, (x5.shape[2], x5.shape[3]))
                # x5 = x5 + x5_pos_map
                
                x6 = self.conv_down6(x5)
                # x6_pos_map = self.pos_encoding(slice_pos, (x6.shape[2], x6.shape[3]))
                # x6 = x6 + x6_pos_map
                
                x7 = self.conv_down7(x6)
                # x7_pos_map = self.pos_encoding(slice_pos, (x7.shape[2], x7.shape[3]))
                # x7 = x7 + x7_pos_map
                
                x8 = self.conv_down8(x7)
                # x8_pos_map = self.pos_encoding(slice_pos, (x8.shape[2], x8.shape[3]))
                # x8 = x8 + x8_pos_map
                
                x8d = self.dropout0(x8)
                # print(x1.shape)
                
                x8d_pos_map = self.pos_encoding(slice_pos, (x8d.shape[2], x8d.shape[3]))
                x8d = x8d + x8d_pos_map
                
                # print(x8d.shape, x7.shape)
                # Decoder 1 - 4 coronaries
                x9 = self.conv_up1(x8d, x7)
                x10 = self.conv_up2(x9, x6)
                x11 = self.conv_up3(x10, x5)
                x12 = self.conv_up4(x11, x4)
                x13 = self.conv_up5(x12, x3)
                x14 = self.conv_up6(x13, x2)
                x15 = self.conv_up7(x14, x1)
                x16 = self.conv_up8(x15, x01r)
                xout = self.conv_double_out(x16)
                
                # Decoder 2 - binary lesion
                x9c = self.conv_up1_class(x8d, torch.cat((x9, x7), dim=1))
                x10c = self.conv_up2_class(x9c, torch.cat((x10, x6), dim=1))

                x11c = self.conv_up3_class(x10c, torch.cat((x11, x5), dim=1))
                x12c = self.conv_up4_class(x11c, torch.cat((x12, x4), dim=1))
                x13c = self.conv_up5_class(x12c, torch.cat((x13, x3), dim=1))
                x14c = self.conv_up6_class(x13c, torch.cat((x14, x2), dim=1))
                x15c = self.conv_up7_class(x14c, torch.cat((x15, x1), dim=1))
                x16c = self.conv_up8_class(x15c, torch.cat((x16, x01r, x), dim=1))

                xoutc1 = self.conv0(x16c)
                xoutc2 = self.relu0(xoutc1)
                xoutc3 = self.conv1(xoutc2)
                # xoutc4 = self.soft(xoutc3)
                Y_region = xout
                Y_lesion = xoutc3
                return Y_region, Y_lesion

        # Create model
        mtal = MTAL(c_pos=c_pos, hidden_dim=hidden_dim)
        mtal.train()
        if self.params['device']=='cuda':
            mtal.cuda() 
        self.mtal=mtal

        # Set default learning rate
        # self.opt_unet_prior = optim.Adam(self.mtal.parameters(), lr = self.params['lr'], betas=(0.9, 0.999), weight_decay=0.01)

    def load(self, modelpath):
        """
        Load pretained model
        """
        self.mtal.load_state_dict(torch.load(modelpath))
        
    def load_checkpoint(self, model_path):
        """
        Load model from checkpoint
        """
        checkpoint = torch.load(model_path)
        self.mtal.load_state_dict(checkpoint['model_state'])

    def predict(self, Xin):
        self.mtal.eval()
        with torch.no_grad():
            Y_region, Y_lesion = self.mtal(Xin)
        return Y_region, Y_lesion
    
    def forward(self, Xin):
        """
        Forward pass
        """
        Y_region, Y_lesion = self.mtal(Xin)
        return Y_region, Y_lesion

