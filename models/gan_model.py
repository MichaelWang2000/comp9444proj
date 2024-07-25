# gan_model.py
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
import numpy as np


# class Discriminator(nn.Module):
#     def __init__(self, input_nc = 3, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d, use_sigmoid=False, getIntermFeat=False):
#         super(Discriminator, self).__init__()
#         self.getIntermFeat = getIntermFeat
#         self.n_layers = n_layers

#         kw = 4
#         padw = int(np.ceil((kw-1.0)/2))
#         sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

#         nf = ndf
#         for n in range(1, n_layers):
#             nf_prev = nf
#             nf = min(nf * 2, 512)
#             sequence += [[
#                 nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
#                 norm_layer(nf), nn.LeakyReLU(0.2, True)
#             ]]

#         nf_prev = nf
#         nf = min(nf * 2, 512)
#         sequence += [[
#             nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
#             norm_layer(nf),
#             nn.LeakyReLU(0.2, True)
#         ]]

#         sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

#         if use_sigmoid:
#             sequence += [[nn.Sigmoid()]]

#         if getIntermFeat:
#             for n in range(len(sequence)):
#                 setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
#         else:
#             sequence_stream = []
#             for n in range(len(sequence)):
#                 sequence_stream += sequence[n]
#             self.model = nn.Sequential(*sequence_stream)

#     def forward(self, input):
#         if self.getIntermFeat:
#             res = [input]
#             for n in range(self.n_layers+2):
#                 model = getattr(self, 'model'+str(n))
#                 res.append(model(res[-1]))
#             return res[1:]
#         else:
#             return self.model(input)  






class WGANGPDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(WGANGPDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0)
        self.bn1 = nn.InstanceNorm2d(128)
        self.bn2 = nn.InstanceNorm2d(256)
        self.bn3 = nn.InstanceNorm2d(512)
        
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.bn1(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn2(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv4(x)), 0.2)
        x = self.conv5(x)
        return x

# # 梯度惩罚函数
# def gradient_penalty(discriminator, real_samples, fake_samples, device='cuda'):
#     batch_size, channels, height, width = real_samples.shape
#     alpha = torch.rand(batch_size, 1, 1, 1).to(device)
#     alpha = alpha.expand_as(real_samples)

#     interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
#     d_interpolates = discriminator(interpolates)
    
#     fake = torch.ones(d_interpolates.size()).to(device)
#     gradients = torch.autograd.grad(
#         outputs=d_interpolates,
#         inputs=interpolates,
#         grad_outputs=fake,
#         create_graph=True,
#         retain_graph=True,
#         only_inputs=True
#     )[0]
#     gradients = gradients.view(batch_size, -1)
#     gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
#     return gradient_penalty











# class Discriminator(nn.Module):
#     def __init__(self, params):
#         super().__init__()

#         # Input Dimension: (nc) x 64 x 64
#         self.conv1 = nn.Conv2d(params['nc'], params['ndf'],
#             4, 2, 1, bias=False)

#         # Input Dimension: (ndf) x 32 x 32
#         self.conv2 = nn.Conv2d(params['ndf'], params['ndf']*2,
#             4, 2, 1, bias=False)
#         self.bn2 = nn.BatchNorm2d(params['ndf']*2)

#         # Input Dimension: (ndf*2) x 16 x 16
#         self.conv3 = nn.Conv2d(params['ndf']*2, params['ndf']*4,
#             4, 2, 1, bias=False)
#         self.bn3 = nn.BatchNorm2d(params['ndf']*4)

#         # Input Dimension: (ndf*4) x 8 x 8
#         self.conv4 = nn.Conv2d(params['ndf']*4, params['ndf']*8,
#             4, 2, 1, bias=False)
#         self.bn4 = nn.BatchNorm2d(params['ndf']*8)

#         # Input Dimension: (ndf*8) x 4 x 4
#         self.conv5 = nn.Conv2d(params['ndf']*8, 1, 4, 1, 0, bias=False)

#     def forward(self, x):
#         x = F.leaky_relu(self.conv1(x), 0.2, True)
#         x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2, True)
#         x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2, True)
#         x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2, True)

#         x = F.sigmoid(self.conv5(x))

#         return x




class Discriminator(nn.Module):
    def __init__(self, params):
        super(Discriminator, self).__init__()

        # Content Image Processing Layers
        self.conv1_content = nn.Conv2d(params['nc'], params['ndf'], 4, 2, 1, bias=False)
        self.conv2_content = nn.Conv2d(params['ndf'], params['ndf']*2, 4, 2, 1, bias=False)
        self.bn2_content = nn.BatchNorm2d(params['ndf']*2)
        self.conv3_content = nn.Conv2d(params['ndf']*2, params['ndf']*4, 4, 2, 1, bias=False)
        self.bn3_content = nn.BatchNorm2d(params['ndf']*4)
        self.conv4_content = nn.Conv2d(params['ndf']*4, params['ndf']*8, 4, 2, 1, bias=False)
        self.bn4_content = nn.BatchNorm2d(params['ndf']*8)
        
        # Style Image Processing Layers
        self.conv1_style = nn.Conv2d(params['nc'], params['ndf'], 4, 2, 1, bias=False)
        self.conv2_style = nn.Conv2d(params['ndf'], params['ndf']*2, 4, 2, 1, bias=False)
        self.bn2_style = nn.BatchNorm2d(params['ndf']*2)
        self.conv3_style = nn.Conv2d(params['ndf']*2, params['ndf']*4, 4, 2, 1, bias=False)
        self.bn3_style = nn.BatchNorm2d(params['ndf']*4)
        self.conv4_style = nn.Conv2d(params['ndf']*4, params['ndf']*8, 4, 2, 1, bias=False)
        self.bn4_style = nn.BatchNorm2d(params['ndf']*8)
        
        # Combined Layer
        self.conv5 = nn.Conv2d(params['ndf']*8, 1, 4, 1, 0, bias=False)

    def forward(self, content, style):
        x_content = F.leaky_relu(self.conv1_content(content), 0.2, True)
        x_content = F.leaky_relu(self.bn2_content(self.conv2_content(x_content)), 0.2, True)
        x_content = F.leaky_relu(self.bn3_content(self.conv3_content(x_content)), 0.2, True)
        x_content = F.leaky_relu(self.bn4_content(self.conv4_content(x_content)), 0.2, True)
        
        x_style = F.leaky_relu(self.conv1_style(style), 0.2, True)
        x_style = F.leaky_relu(self.bn2_style(self.conv2_style(x_style)), 0.2, True)
        x_style = F.leaky_relu(self.bn3_style(self.conv3_style(x_style)), 0.2, True)
        x_style = F.leaky_relu(self.bn4_style(self.conv4_style(x_style)), 0.2, True)
        
        # Combine content and style features
        x = x_content + x_style
        x = torch.sigmoid(self.conv5(x))
        
        return x











# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         self.main = nn.Sequential(
#             nn.Conv2d(3, 64, 4, 2, 1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(64, 128, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(128, 256, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(256, 512, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(512, 1024, 4, 2, 1, bias=False), 
#             nn.BatchNorm2d(1024),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(1024, 1, 4, 1, 0, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, input):
#         return self.main(input).view(-1, 1).squeeze(1)

# import torch
# import torch.nn as nn
# from torch.nn.utils import spectral_norm

# def init_weights(m):
#     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
#         nn.init.xavier_uniform_(m.weight)
#         if m.bias is not None:
#             nn.init.constant_(m.bias, 0)

# class Discriminator(nn.Module):
#     def __init__(self, nc=3, ndf=64):
#         super(Discriminator, self).__init__()
#         self.nc = nc
#         self.output_dim = 1

#         self.conv1 = nn.Sequential(
#             # 256->128
#             spectral_norm(nn.Conv2d(self.nc, ndf, 4, 2, 1, bias=False)),
#             nn.LeakyReLU(0.2, inplace=True),
#         )

#         self.skip_out1 = nn.Sequential(
#             nn.Conv2d(ndf, 32, 3, 1, 1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.AdaptiveAvgPool2d(8),
#             nn.Conv2d(32, self.output_dim, 3, 1, 1, bias=False)
#         )
        
#         self.conv2 = nn.Sequential(
#             # 128->64
#             spectral_norm(nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False)),
#             nn.InstanceNorm2d(ndf*2),
#             nn.LeakyReLU(0.2, inplace=True),
#         )
        
#         self.conv3 = nn.Sequential(
#             # 64->32
#             spectral_norm(nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False)),
#             nn.InstanceNorm2d(ndf*4),
#             nn.LeakyReLU(0.2, inplace=True),
#         )

#         self.skip_out2 = nn.Sequential(
#             nn.Conv2d(ndf*4, 32, 3, 1, 1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.AdaptiveAvgPool2d(8),
#             nn.Conv2d(32, self.output_dim, 3, 1, 1, bias=False)
#         )
        
#         self.conv4 = nn.Sequential(
#             # 32->16
#             spectral_norm(nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False)),
#             nn.InstanceNorm2d(ndf*8),
#             nn.LeakyReLU(0.2, inplace=True),
#         )

#         self.conv5 = nn.Sequential(
#             # 16->8
#             spectral_norm(nn.Conv2d(ndf*8, self.output_dim, 4, 2, 1, bias=False)),
#         )

#         init_weights(self)

#     def forward(self, input):
#         out = self.conv1(input)
#         skip_out1 = self.skip_out1(out)
#         out = self.conv3(self.conv2(out))
#         skip_out2 = self.skip_out2(out)
#         out = self.conv5(self.conv4(out))
#         out = ((out + skip_out1 + skip_out2) * 1 / 3).squeeze()
#         return out