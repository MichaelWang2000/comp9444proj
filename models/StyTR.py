import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
from function import normal,normal_style
from function import calc_mean_std
import scipy.stats as stats
from models.ViT_helper import DropPath, to_2tuple, trunc_normal_

"""
Tommy's improvement
"""
# def calc_mean_std(feat, eps=1e-5):
#     # eps 是一个小值，防止除零
#     size = feat.size()
#     assert (len(size) == 4)
#     N, C = size[:2]
#     feat_var = feat.view(N, C, -1).var(dim=2) + eps
#     feat_std = feat_var.sqrt().view(N, C, 1, 1)
#     feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
#     return feat_mean, feat_std

# class PatternRepeatability(nn.Module):
#     def __init__(self):
#         super(PatternRepeatability, self).__init__()

#     def forward(self, input):
#         B, C, H, W = input.size()
#         input_mean, input_std = calc_mean_std(input)
#         normalized_input = (input - input_mean) / input_std
#         return normalized_input

# class MultiScaleImageDiscriminator(nn.Module):
#     def __init__(self, nc=3, ndf=64):
#         super(MultiScaleImageDiscriminator, self).__init__()
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
#             # nn.BatchNorm2d(ndf*2),
#             nn.InstanceNorm2d(ndf*2),
#             nn.LeakyReLU(0.2, inplace=True),
#         )
        
#         self.conv3 = nn.Sequential(
#             # 64->32
#             spectral_norm(nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False)),
#             # nn.BatchNorm2d(ndf*4),
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
#             # nn.BatchNorm2d(ndf*8),
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
#         out = ((out + skip_out1 + skip_out2) * 1/3).squeeze()
#         # out = ((out + skip_out1 + skip_out2)).squeeze()
        
#         return out





# class MultiScaleFeatureFusion(nn.Module):
#     def __init__(self, in_planes):
#         super(MultiScaleFeatureFusion, self).__init__()
#         self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=1, stride=1, padding=0)
#         self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1)
#         self.conv3 = nn.Conv2d(in_planes, in_planes, kernel_size=5, stride=1, padding=2)
#         self.conv4 = nn.Conv2d(in_planes, in_planes, kernel_size=7, stride=1, padding=3)
#         self.final_conv = nn.Conv2d(in_planes * 4, in_planes, kernel_size=1, stride=1, padding=0)
    
#     def forward(self, x):
#         scale1 = self.conv1(x)
#         scale2 = self.conv2(x)
#         scale3 = self.conv3(x)
#         scale4 = self.conv4(x)
        
#         # 将不同尺度的特征连接起来
#         out = torch.cat((scale1, scale2, scale3, scale4), dim=1)
#         out = self.final_conv(out)
#         return out

# class AdaptiveMultiScaleAttention(nn.Module):
#     def __init__(self, in_planes, out_planes, query_planes=None, key_planes=None):
#         super(AdaptiveMultiScaleAttention, self).__init__()
#         if key_planes is None:
#             key_planes = in_planes
#         if query_planes is None:
#             query_planes = in_planes
        
#         self.query_conv = nn.Conv2d(query_planes, key_planes, kernel_size=1)
#         self.key_conv = nn.Conv2d(key_planes, key_planes, kernel_size=1)
#         self.value_conv = nn.Conv2d(in_planes, out_planes, kernel_size=1)
#         self.softmax = nn.Softmax(dim=-1)
#         self.out_conv = nn.Conv2d(out_planes, out_planes, kernel_size=1)

#     def forward(self, content, style, content_key, style_key):
#         B, C, H, W = content.size()

#         query = self.query_conv(content_key).view(B, -1, H * W).permute(0, 2, 1)  # B x HW x C
#         key = self.key_conv(style_key).view(B, -1, H * W)  # B x C x HW
#         value = self.value_conv(style).view(B, -1, H * W).permute(0, 2, 1)  # B x HW x C

#         attention = self.softmax(torch.bmm(query, key))  # B x HW x HW
#         out = torch.bmm(attention, value).permute(0, 2, 1).view(B, C, H, W)  # B x C x H x W

#         out = self.out_conv(out)
#         return out






"""
Original version
"""
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=256, patch_size=8, in_chans=3, embed_dim=512):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)

        return x


decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
class StyTrans(nn.Module):
    """ This is the style transform transformer module """
    
    def __init__(self,encoder,decoder,PatchEmbed, transformer,args):

        super().__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.enc_5 = nn.Sequential(*enc_layers[31:44])  # relu4_1 -> relu5_1
        
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4', 'enc_5']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

        self.mse_loss = nn.MSELoss()
        self.transformer = transformer
        hidden_dim = transformer.d_model       
        self.decode = decoder
        self.embedding = PatchEmbed
        self.pattern_repeatability = PatternRepeatability()
        #self.multi_scale_fusion = MultiScaleFeatureFusion(hidden_dim)
        # self.adaptive_attention = AdaptiveMultiScaleAttention(hidden_dim, hidden_dim)

    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(5):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    def calc_content_loss(self, input, target):
      assert (input.size() == target.size())
      assert (target.requires_grad is False)
      return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)
    def forward(self, samples_c: NestedTensor,samples_s: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        """
        content_input = samples_c
        style_input = samples_s
        if isinstance(samples_c, (list, torch.Tensor)):
            samples_c = nested_tensor_from_tensor_list(samples_c)   # support different-sized images padding is used for mask [tensor, mask] 
        if isinstance(samples_s, (list, torch.Tensor)):
            samples_s = nested_tensor_from_tensor_list(samples_s) 
        
        # ### features used to calcate loss 
        content_feats = self.encode_with_intermediate(samples_c.tensors)
        style_feats = self.encode_with_intermediate(samples_s.tensors)

        ### Linear projection
        style = self.embedding(samples_s.tensors)
        content = self.embedding(samples_c.tensors)
        
        # postional embedding is calculated in transformer.py
        pos_s = None
        pos_c = None

        mask = None
        hs = self.transformer(style, mask , content, pos_c, pos_s)
        # 使用多尺度特征融合
        #hs = self.multi_scale_fusion(hs)
         # 使用自适应多尺度注意力机制
        # hs = self.adaptive_attention(content, style, content_feats[-1], style_feats[-1])
        Ics = self.decode(hs)

        Ics_feats = self.encode_with_intermediate(Ics)
        loss_c = self.calc_content_loss(normal(Ics_feats[-1]), normal(content_feats[-1]))+self.calc_content_loss(normal(Ics_feats[-2]), normal(content_feats[-2]))
        # Style loss
        loss_s = self.calc_style_loss(Ics_feats[0], style_feats[0])
        for i in range(1, 5):
            loss_s += self.calc_style_loss(Ics_feats[i], style_feats[i])
            
        
        Icc = self.decode(self.transformer(content, mask , content, pos_c, pos_c))
        Iss = self.decode(self.transformer(style, mask , style, pos_s, pos_s))    

        #Identity losses lambda 1    
        loss_lambda1 = self.calc_content_loss(Icc,content_input)+self.calc_content_loss(Iss,style_input)
        
        #Identity losses lambda 2
        Icc_feats=self.encode_with_intermediate(Icc)
        Iss_feats=self.encode_with_intermediate(Iss)
        loss_lambda2 = self.calc_content_loss(Icc_feats[0], content_feats[0])+self.calc_content_loss(Iss_feats[0], style_feats[0])
        for i in range(1, 5):
            loss_lambda2 += self.calc_content_loss(Icc_feats[i], content_feats[i])+self.calc_content_loss(Iss_feats[i], style_feats[i])
        # Please select and comment out one of the following two sentences
        return Ics,  loss_c, loss_s, loss_lambda1, loss_lambda2   #train
        # return Ics    #test 


"""
no_use_upsample
"""
# import torch
# import torch.nn.functional as F
# from torch import nn
# import numpy as np
# from util import box_ops
# from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
#                        accuracy, get_world_size, interpolate,
#                        is_dist_avail_and_initialized)
# from function import normal,normal_style
# from function import calc_mean_std
# import scipy.stats as stats
# from models.ViT_helper import DropPath, to_2tuple, trunc_normal_

# class PatchEmbed(nn.Module):
#     """ Image to Patch Embedding """
#     def __init__(self, img_size=256, patch_size=8, in_chans=3, embed_dim=512):
#         super().__init__()
#         img_size = to_2tuple(img_size)
#         patch_size = to_2tuple(patch_size)
#         num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.num_patches = num_patches
        
#         self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
#         self.downsample = nn.Upsample(scale_factor=1/patch_size[0], mode='nearest')  # Downsample

#     def forward(self, x):
#         B, C, H, W = x.shape
#         x = self.proj(x)
#         x = self.downsample(x)
#         return x



# decoder = nn.Sequential(
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(512, 256, (3, 3)),
#     nn.ReLU(),
#     nn.Upsample(scale_factor=2, mode='nearest'),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(256, 256, (3, 3)),
#     nn.ReLU(),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(256, 256, (3, 3)),
#     nn.ReLU(),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(256, 256, (3, 3)),
#     nn.ReLU(),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(256, 128, (3, 3)),
#     nn.ReLU(),
#     nn.Upsample(scale_factor=2, mode='nearest'),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(128, 128, (3, 3)),
#     nn.ReLU(),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(128, 64, (3, 3)),
#     nn.ReLU(),
#     nn.Upsample(scale_factor=2, mode='nearest'),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(64, 64, (3, 3)),
#     nn.ReLU(),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(64, 3, (3, 3)),
# )

# vgg = nn.Sequential(
#     nn.Conv2d(3, 3, (1, 1)),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(3, 64, (3, 3)),
#     nn.ReLU(),  # relu1-1
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(64, 64, (3, 3)),
#     nn.ReLU(),  # relu1-2
#     nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(64, 128, (3, 3)),
#     nn.ReLU(),  # relu2-1
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(128, 128, (3, 3)),
#     nn.ReLU(),  # relu2-2
#     nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(128, 256, (3, 3)),
#     nn.ReLU(),  # relu3-1
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(256, 256, (3, 3)),
#     nn.ReLU(),  # relu3-2
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(256, 256, (3, 3)),
#     nn.ReLU(),  # relu3-3
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(256, 256, (3, 3)),
#     nn.ReLU(),  # relu3-4
#     nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(256, 512, (3, 3)),
#     nn.ReLU(),  # relu4-1, this is the last layer used
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(512, 512, (3, 3)),
#     nn.ReLU(),  # relu4-2
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(512, 512, (3, 3)),
#     nn.ReLU(),  # relu4-3
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(512, 512, (3, 3)),
#     nn.ReLU(),  # relu4-4
#     nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(512, 512, (3, 3)),
#     nn.ReLU(),  # relu5-1
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(512, 512, (3, 3)),
#     nn.ReLU(),  # relu5-2
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(512, 512, (3, 3)),
#     nn.ReLU(),  # relu5-3
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(512, 512, (3, 3)),
#     nn.ReLU()  # relu5-4
# )

# class MLP(nn.Module):
#     """ Very simple multi-layer perceptron (also called FFN)"""

#     def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
#         super().__init__()
#         self.num_layers = num_layers
#         h = [hidden_dim] * (num_layers - 1)
#         self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

#     def forward(self, x):
#         for i, layer in enumerate(self.layers):
#             x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
#         return x

# class StyTrans(nn.Module):
#     """ This is the style transform transformer module """
    
#     def __init__(self, encoder, decoder, PatchEmbed, transformer, args):
#         super().__init__()
#         enc_layers = list(encoder.children())
#         self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
#         self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
#         self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
#         self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
#         self.enc_5 = nn.Sequential(*enc_layers[31:44])  # relu4_1 -> relu5_1
        
#         for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4', 'enc_5']:
#             for param in getattr(self, name).parameters():
#                 param.requires_grad = False

#         self.mse_loss = nn.MSELoss()
#         self.transformer = transformer
#         hidden_dim = transformer.d_model       
#         self.decode = decoder
#         self.embedding = PatchEmbed
#         self.upsample = nn.Upsample(scale_factor=PatchEmbed.patch_size[0], mode='nearest')  # Upsample

#     def encode_with_intermediate(self, input):
#         results = [input]
#         for i in range(5):
#             func = getattr(self, 'enc_{:d}'.format(i + 1))
#             results.append(func(results[-1]))
#         return results[1:]

#     def calc_content_loss(self, input, target):
#         assert (input.size() == target.size()), f"Input size: {input.size()}, Target size: {target.size()}"
#         assert (target.requires_grad is False)
#         return self.mse_loss(input, target)

#     def calc_style_loss(self, input, target):
#         assert (input.size() == target.size())
#         assert (target.requires_grad is False)
#         input_mean, input_std = calc_mean_std(input)
#         target_mean, target_std = calc_mean_std(target)
#         return self.mse_loss(input_mean, target_mean) + \
#                self.mse_loss(input_std, target_std)
    
#     def forward(self, samples_c: NestedTensor, samples_s: NestedTensor):
#         """ The forward expects a NestedTensor, which consists of:
#                - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
#                - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
#         """
#         content_input = samples_c
#         style_input = samples_s
#         if isinstance(samples_c, (list, torch.Tensor)):
#             samples_c = nested_tensor_from_tensor_list(samples_c)   # support different-sized images padding is used for mask [tensor, mask] 
#         if isinstance(samples_s, (list, torch.Tensor)):
#             samples_s = nested_tensor_from_tensor_list(samples_s) 
        
#         # Features used to calculate loss 
#         content_feats = self.encode_with_intermediate(samples_c.tensors)
#         style_feats = self.encode_with_intermediate(samples_s.tensors)

#         # Linear projection
#         style = self.embedding(samples_s.tensors)
#         content = self.embedding(samples_c.tensors)
        
#         # Positional embedding is calculated in transformer.py
#         pos_s = None
#         pos_c = None

#         mask = None
#         hs = self.transformer(style, mask, content, pos_c, pos_s)   
        
#         # Upsample before passing to the decoder
#         hs = self.upsample(hs)
#         Ics = self.decode(hs)

#         Ics_feats = self.encode_with_intermediate(Ics)
        
#         # Debugging output sizes
#         print(f"Content feature size at last layer: {content_feats[-1].size()}")
#         print(f"Generated image feature size at last layer: {Ics_feats[-1].size()}")

#         loss_c = self.calc_content_loss(normal(Ics_feats[-1]), normal(content_feats[-1])) + \
#                  self.calc_content_loss(normal(Ics_feats[-2]), normal(content_feats[-2]))
#         # Style loss
#         loss_s = self.calc_style_loss(Ics_feats[0], style_feats[0])
#         for i in range(1, 5):
#             loss_s += self.calc_style_loss(Ics_feats[i], style_feats[i])
            
#         Icc = self.decode(self.upsample(self.transformer(content, mask, content, pos_c, pos_c)))
#         Iss = self.decode(self.upsample(self.transformer(style, mask, style, pos_s, pos_s)))    

#         # Identity losses lambda 1    
#         loss_lambda1 = self.calc_content_loss(Icc, content_input) + self.calc_content_loss(Iss, style_input)
        
#         # Identity losses lambda 2
#         Icc_feats = self.encode_with_intermediate(Icc)
#         Iss_feats = self.encode_with_intermediate(Iss)
#         loss_lambda2 = self.calc_content_loss(Icc_feats[0], content_feats[0]) + \
#                        self.calc_content_loss(Iss_feats[0], style_feats[0])
#         for i in range(1, 5):
#             loss_lambda2 += self.calc_content_loss(Icc_feats[i], content_feats[i]) + \
#                             self.calc_content_loss(Iss_feats[i], style_feats[i])
#         # Please select and comment out one of the following two sentences
#         return Ics,  loss_c, loss_s, loss_lambda1, loss_lambda2   #train
#         # return Ics    #test



"""
try_bilinear
"""

# import torch
# import torch.nn.functional as F
# from torch import nn
# import numpy as np
# from util import box_ops
# from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
#                        accuracy, get_world_size, interpolate,
#                        is_dist_avail_and_initialized)
# from function import normal,normal_style
# from function import calc_mean_std
# import scipy.stats as stats
# from models.ViT_helper import DropPath, to_2tuple, trunc_normal_

# class PatchEmbed(nn.Module):
#     """ Image to Patch Embedding
#     """
#     def __init__(self, img_size=256, patch_size=8, in_chans=3, embed_dim=512):
#         super().__init__()
#         img_size = to_2tuple(img_size)
#         patch_size = to_2tuple(patch_size)
#         num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.num_patches = num_patches
        
#         self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
#         #self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
#         self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    

#     def forward(self, x):
#         B, C, H, W = x.shape
#         x = self.proj(x)

#         return x


# decoder = nn.Sequential(
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(512, 256, (3, 3)),
#     nn.ReLU(),
#     #nn.Upsample(scale_factor=2, mode='nearest'),
#     nn.Upsample(scale_factor=2, mode='nearest'),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(256, 256, (3, 3)),
#     nn.ReLU(),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(256, 256, (3, 3)),
#     nn.ReLU(),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(256, 256, (3, 3)),
#     nn.ReLU(),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(256, 128, (3, 3)),
#     nn.ReLU(),
#     #nn.Upsample(scale_factor=2, mode='nearest'),
#     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(128, 128, (3, 3)),
#     nn.ReLU(),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(128, 64, (3, 3)),
#     nn.ReLU(),
#     #nn.Upsample(scale_factor=2, mode='nearest'),
#     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(64, 64, (3, 3)),
#     nn.ReLU(),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(64, 3, (3, 3)),
# )

# vgg = nn.Sequential(
#     nn.Conv2d(3, 3, (1, 1)),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(3, 64, (3, 3)),
#     nn.ReLU(),  # relu1-1
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(64, 64, (3, 3)),
#     nn.ReLU(),  # relu1-2
#     nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(64, 128, (3, 3)),
#     nn.ReLU(),  # relu2-1
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(128, 128, (3, 3)),
#     nn.ReLU(),  # relu2-2
#     nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(128, 256, (3, 3)),
#     nn.ReLU(),  # relu3-1
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(256, 256, (3, 3)),
#     nn.ReLU(),  # relu3-2
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(256, 256, (3, 3)),
#     nn.ReLU(),  # relu3-3
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(256, 256, (3, 3)),
#     nn.ReLU(),  # relu3-4
#     nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(256, 512, (3, 3)),
#     nn.ReLU(),  # relu4-1, this is the last layer used
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(512, 512, (3, 3)),
#     nn.ReLU(),  # relu4-2
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(512, 512, (3, 3)),
#     nn.ReLU(),  # relu4-3
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(512, 512, (3, 3)),
#     nn.ReLU(),  # relu4-4
#     nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(512, 512, (3, 3)),
#     nn.ReLU(),  # relu5-1
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(512, 512, (3, 3)),
#     nn.ReLU(),  # relu5-2
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(512, 512, (3, 3)),
#     nn.ReLU(),  # relu5-3
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(512, 512, (3, 3)),
#     nn.ReLU()  # relu5-4
# )

# class MLP(nn.Module):
#     """ Very simple multi-layer perceptron (also called FFN)"""

#     def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
#         super().__init__()
#         self.num_layers = num_layers
#         h = [hidden_dim] * (num_layers - 1)
#         self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

#     def forward(self, x):
#         for i, layer in enumerate(self.layers):
#             x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
#         return x
# class StyTrans(nn.Module):
#     """ This is the style transform transformer module """
    
#     def __init__(self,encoder,decoder,PatchEmbed, transformer,args):

#         super().__init__()
#         enc_layers = list(encoder.children())
#         self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
#         self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
#         self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
#         self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
#         self.enc_5 = nn.Sequential(*enc_layers[31:44])  # relu4_1 -> relu5_1
        
#         for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4', 'enc_5']:
#             for param in getattr(self, name).parameters():
#                 param.requires_grad = False

#         self.mse_loss = nn.MSELoss()
#         self.transformer = transformer
#         hidden_dim = transformer.d_model       
#         self.decode = decoder
#         self.embedding = PatchEmbed

#     def encode_with_intermediate(self, input):
#         results = [input]
#         for i in range(5):
#             func = getattr(self, 'enc_{:d}'.format(i + 1))
#             results.append(func(results[-1]))
#         return results[1:]

#     def calc_content_loss(self, input, target):
#       assert (input.size() == target.size())
#       assert (target.requires_grad is False)
#       return self.mse_loss(input, target)

#     def calc_style_loss(self, input, target):
#         assert (input.size() == target.size())
#         assert (target.requires_grad is False)
#         input_mean, input_std = calc_mean_std(input)
#         target_mean, target_std = calc_mean_std(target)
#         return self.mse_loss(input_mean, target_mean) + \
#                self.mse_loss(input_std, target_std)
#     def forward(self, samples_c: NestedTensor,samples_s: NestedTensor):
#         """ The forward expects a NestedTensor, which consists of:
#                - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
#                - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

#         """
#         content_input = samples_c
#         style_input = samples_s
#         if isinstance(samples_c, (list, torch.Tensor)):
#             samples_c = nested_tensor_from_tensor_list(samples_c)   # support different-sized images padding is used for mask [tensor, mask] 
#         if isinstance(samples_s, (list, torch.Tensor)):
#             samples_s = nested_tensor_from_tensor_list(samples_s) 
        
#         # ### features used to calcate loss 
#         content_feats = self.encode_with_intermediate(samples_c.tensors)
#         style_feats = self.encode_with_intermediate(samples_s.tensors)

#         ### Linear projection
#         style = self.embedding(samples_s.tensors)
#         content = self.embedding(samples_c.tensors)
        
#         # postional embedding is calculated in transformer.py
#         pos_s = None
#         pos_c = None

#         mask = None
#         hs = self.transformer(style, mask , content, pos_c, pos_s)   
#         Ics = self.decode(hs)

#         Ics_feats = self.encode_with_intermediate(Ics)
#         loss_c = self.calc_content_loss(normal(Ics_feats[-1]), normal(content_feats[-1]))+self.calc_content_loss(normal(Ics_feats[-2]), normal(content_feats[-2]))
#         # Style loss
#         loss_s = self.calc_style_loss(Ics_feats[0], style_feats[0])
#         for i in range(1, 5):
#             loss_s += self.calc_style_loss(Ics_feats[i], style_feats[i])
            
        
#         Icc = self.decode(self.transformer(content, mask , content, pos_c, pos_c))
#         Iss = self.decode(self.transformer(style, mask , style, pos_s, pos_s))    

#         #Identity losses lambda 1    
#         loss_lambda1 = self.calc_content_loss(Icc,content_input)+self.calc_content_loss(Iss,style_input)
        
#         #Identity losses lambda 2
#         Icc_feats=self.encode_with_intermediate(Icc)
#         Iss_feats=self.encode_with_intermediate(Iss)
#         loss_lambda2 = self.calc_content_loss(Icc_feats[0], content_feats[0])+self.calc_content_loss(Iss_feats[0], style_feats[0])
#         for i in range(1, 5):
#             loss_lambda2 += self.calc_content_loss(Icc_feats[i], content_feats[i])+self.calc_content_loss(Iss_feats[i], style_feats[i])
#         # Please select and comment out one of the following two sentences
#         return Ics,  loss_c, loss_s, loss_lambda1, loss_lambda2   #train
#         # return Ics    #test 

# class StyTrans(nn.Module):
#     def __init__(self, encoder, decoder, PatchEmbed, transformer, args):
#         super().__init__()
#         enc_layers = list(encoder.children())
#         self.enc_1 = nn.Sequential(*enc_layers[:4])
#         self.enc_2 = nn.Sequential(*enc_layers[4:11])
#         self.enc_3 = nn.Sequential(*enc_layers[11:18])
#         self.enc_4 = nn.Sequential(*enc_layers[18:31])
#         self.enc_5 = nn.Sequential(*enc_layers[31:44])
        
#         for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4', 'enc_5']:
#             for param in getattr(self, name).parameters():
#                 param.requires_grad = False

#         self.mse_loss = nn.MSELoss()
#         self.transformer = transformer
#         hidden_dim = transformer.d_model       
#         self.decode = decoder
#         self.embedding = PatchEmbed
#         self.upsample = nn.Upsample(scale_factor=PatchEmbed.patch_size[0], mode='nearest')

#     def encode_with_intermediate(self, input):
#         results = [input]
#         for i in range(5):
#             func = getattr(self, 'enc_{:d}'.format(i + 1))
#             results.append(func(results[-1]))
#         return results[1:]

#     def calc_content_loss(self, input, target):
#         assert (input.size() == target.size()), f"Input size: {input.size()}, Target size: {target.size()}"
#         assert (target.requires_grad is False)
#         return self.mse_loss(input, target)

#     def calc_style_loss(self, input, target):
#         assert (input.size() == target.size())
#         assert (target.requires_grad is False)
#         input_mean, input_std = calc_mean_std(input)
#         target_mean, target_std = calc_mean_std(target)
#         return self.mse_loss(input_mean, target_mean) + \
#                self.mse_loss(input_std, target_std)
    
#     def forward(self, samples_c: NestedTensor, samples_s: NestedTensor):
#         """ The forward expects a NestedTensor, which consists of:
#                - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
#                - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
#         """
#         content_input = samples_c
#         style_input = samples_s
#         if isinstance(samples_c, (list, torch.Tensor)):
#             samples_c = nested_tensor_from_tensor_list(samples_c)   # support different-sized images padding is used for mask [tensor, mask] 
#         if isinstance(samples_s, (list, torch.Tensor)):
#             samples_s = nested_tensor_from_tensor_list(samples_s) 
        
#         # Features used to calculate loss 
#         content_feats = self.encode_with_intermediate(samples_c.tensors)
#         style_feats = self.encode_with_intermediate(samples_s.tensors)

#         # Linear projection and downsample
#         style = self.embedding(samples_s.tensors)
#         content = self.embedding(samples_c.tensors)
        
#         # Positional embedding is calculated in transformer.py
#         pos_s = None
#         pos_c = None

#         mask = None
#         hs = self.transformer(style, mask, content, pos_c, pos_s)   
        
#         # Upsample before passing to the decoder
#         hs = self.upsample(hs)
#         Ics = self.decode(hs)

#         Ics_feats = self.encode_with_intermediate(Ics)
        
#         # Debugging output sizes
#         print(f"Content feature size at last layer: {content_feats[-1].size()}")
#         print(f"Generated image feature size at last layer: {Ics_feats[-1].size()}")

#         loss_c = self.calc_content_loss(normal(Ics_feats[-1]), normal(content_feats[-1])) + \
#                  self.calc_content_loss(normal(Ics_feats[-2]), normal(content_feats[-2]))
#         # Style loss
#         loss_s = self.calc_style_loss(Ics_feats[0], style_feats[0])
#         for i in range(1, 5):
#             loss_s += self.calc_style_loss(Ics_feats[i], style_feats[i])
            
#         Icc = self.decode(self.upsample(self.transformer(content, mask, content, pos_c, pos_c)))
#         Iss = self.decode(self.upsample(self.transformer(style, mask, style, pos_s, pos_s)))    

#         # Identity losses lambda 1    
#         loss_lambda1 = self.calc_content_loss(Icc, content_input) + self.calc_content_loss(Iss, style_input)
        
#         # Identity losses lambda 2
#         Icc_feats = self.encode_with_intermediate(Icc)
#         Iss_feats = self.encode_with_intermediate(Iss)
#         loss_lambda2 = self.calc_content_loss(Icc_feats[0], content_feats[0]) + \
#                        self.calc_content_loss(Iss_feats[0], style_feats[0])
#         for i in range(1, 5):
#             loss_lambda2 += self.calc_content_loss(Icc_feats[i], content_feats[i]) + \
#                             self.calc_content_loss(Iss_feats[i], style_feats[i])
#         # Please select and comment out one of the following two sentences
#         return Ics,  loss_c, loss_s, loss_lambda1, loss_lambda2   #train
#         # return Ics    #test

"""
no_upsample——best (no_yanse)
"""
# import torch
# import torch.nn.functional as F
# from torch import nn
# import numpy as np
# from util import box_ops
# from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
#                        accuracy, get_world_size, interpolate,
#                        is_dist_avail_and_initialized)
# from function import normal, normal_style
# from function import calc_mean_std
# import scipy.stats as stats
# from models.ViT_helper import DropPath, to_2tuple, trunc_normal_

# class PatchEmbed(nn.Module):
#     """ Image to Patch Embedding
#     """
#     def __init__(self, img_size=256, patch_size=8, in_chans=3, embed_dim=512):
#         super().__init__()
#         img_size = to_2tuple(img_size)
#         patch_size = to_2tuple(patch_size)
#         num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.num_patches = num_patches
        
#         self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
#         self.up1 = nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2)

#     def forward(self, x):
#         B, C, H, W = x.shape
#         x = self.proj(x)
#         return x


# decoder = nn.Sequential(
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(512, 256, (3, 3)),
#     nn.ReLU(),
#     nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(256, 256, (3, 3)),
#     nn.ReLU(),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(256, 256, (3, 3)),
#     nn.ReLU(),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(256, 256, (3, 3)),
#     nn.ReLU(),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(256, 128, (3, 3)),
#     nn.ReLU(),
#     nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(128, 128, (3, 3)),
#     nn.ReLU(),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(128, 64, (3, 3)),
#     nn.ReLU(),
#     nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(64, 64, (3, 3)),
#     nn.ReLU(),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(64, 3, (3, 3)),
# )

# vgg = nn.Sequential(
#     nn.Conv2d(3, 3, (1, 1)),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(3, 64, (3, 3)),
#     nn.ReLU(),  # relu1-1
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(64, 64, (3, 3)),
#     nn.ReLU(),  # relu1-2
#     nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(64, 128, (3, 3)),
#     nn.ReLU(),  # relu2-1
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(128, 128, (3, 3)),
#     nn.ReLU(),  # relu2-2
#     nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(128, 256, (3, 3)),
#     nn.ReLU(),  # relu3-1
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(256, 256, (3, 3)),
#     nn.ReLU(),  # relu3-2
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(256, 256, (3, 3)),
#     nn.ReLU(),  # relu3-3
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(256, 256, (3, 3)),
#     nn.ReLU(),  # relu3-4
#     nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(256, 512, (3, 3)),
#     nn.ReLU(),  # relu4-1, this is the last layer used
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(512, 512, (3, 3)),
#     nn.ReLU(),  # relu4-2
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(512, 512, (3, 3)),
#     nn.ReLU(),  # relu4-3
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(512, 512, (3, 3)),
#     nn.ReLU(),  # relu4-4
#     nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(512, 512, (3, 3)),
#     nn.ReLU(),  # relu5-1
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(512, 512, (3, 3)),
#     nn.ReLU(),  # relu5-2
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(512, 512, (3, 3)),
#     nn.ReLU(),  # relu5-3
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(512, 512, (3, 3)),
#     nn.ReLU()  # relu5-4
# )

# class MLP(nn.Module):
#     """ Very simple multi-layer perceptron (also called FFN)"""

#     def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
#         super().__init__()
#         self.num_layers = num_layers
#         h = [hidden_dim] * (num_layers - 1)
#         self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

#     def forward(self, x):
#         for i, layer in enumerate(self.layers):
#             x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
#         return x

# class StyTrans(nn.Module):
#     """ This is the style transform transformer module """
    
#     def __init__(self, encoder, decoder, PatchEmbed, transformer, args):
#         super().__init__()
#         enc_layers = list(encoder.children())
#         self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
#         self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
#         self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
#         self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
#         self.enc_5 = nn.Sequential(*enc_layers[31:44])  # relu4_1 -> relu5_1
        
#         for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4', 'enc_5']:
#             for param in getattr(self, name).parameters():
#                 param.requires_grad = False

#         self.mse_loss = nn.MSELoss()
#         self.transformer = transformer
#         hidden_dim = transformer.d_model       
#         self.decode = decoder
#         self.embedding = PatchEmbed

#     def encode_with_intermediate(self, input):
#         results = [input]
#         for i in range(5):
#             func = getattr(self, 'enc_{:d}'.format(i + 1))
#             results.append(func(results[-1]))
#         return results[1:]

#     def calc_content_loss(self, input, target):
#         assert (input.size() == target.size())
#         assert (target.requires_grad is False)
#         return self.mse_loss(input, target)

#     def calc_style_loss(self, input, target):
#         assert (input.size() == target.size())
#         assert (target.requires_grad is False)
#         input_mean, input_std = calc_mean_std(input)
#         target_mean, target_std = calc_mean_std(target)
#         return self.mse_loss(input_mean, target_mean) + \
#                self.mse_loss(input_std, target_std)

#     def forward(self, samples_c: NestedTensor, samples_s: NestedTensor):
#         """ The forward expects a NestedTensor, which consists of:
#                - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
#                - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
#         """
#         content_input = samples_c
#         style_input = samples_s
#         if isinstance(samples_c, (list, torch.Tensor)):
#             samples_c = nested_tensor_from_tensor_list(samples_c)   # support different-sized images padding is used for mask [tensor, mask] 
#         if isinstance(samples_s, (list, torch.Tensor)):
#             samples_s = nested_tensor_from_tensor_list(samples_s) 
        
#         # features used to calculate loss 
#         content_feats = self.encode_with_intermediate(samples_c.tensors)
#         style_feats = self.encode_with_intermediate(samples_s.tensors)

#         # Linear projection
#         style = self.embedding(samples_s.tensors)
#         content = self.embedding(samples_c.tensors)
        
#         # positional embedding is calculated in transformer.py
#         pos_s = None
#         pos_c = None

#         mask = None
#         hs = self.transformer(style, mask, content, pos_c, pos_s)   
#         Ics = self.decode(hs)

#         Ics_feats = self.encode_with_intermediate(Ics)
#         loss_c = self.calc_content_loss(normal(Ics_feats[-1]), normal(content_feats[-1])) + self.calc_content_loss(normal(Ics_feats[-2]), normal(content_feats[-2]))
#         # Style loss
#         loss_s = self.calc_style_loss(Ics_feats[0], style_feats[0])
#         for i in range(1, 5):
#             loss_s += self.calc_style_loss(Ics_feats[i], style_feats[i])
        
#         Icc = self.decode(self.transformer(content, mask, content, pos_c, pos_c))
#         Iss = self.decode(self.transformer(style, mask, style, pos_s, pos_s))    

#         # Identity losses lambda 1    
#         loss_lambda1 = self.calc_content_loss(Icc, content_input) + self.calc_content_loss(Iss, style_input)
        
#         # Identity losses lambda 2
#         Icc_feats = self.encode_with_intermediate(Icc)
#         Iss_feats = self.encode_with_intermediate(Iss)
#         loss_lambda2 = self.calc_content_loss(Icc_feats[0], content_feats[0]) + self.calc_content_loss(Iss_feats[0], style_feats[0])
#         for i in range(1, 5):
#             loss_lambda2 += self.calc_content_loss(Icc_feats[i], content_feats[i]) + self.calc_content_loss(Iss_feats[i], style_feats[i])

#         # Please select and comment out one of the following two sentences
#         return Ics,  loss_c, loss_s, loss_lambda1, loss_lambda2   # train
#         # return Ics    # test


"""
new_try_1
"""
# import torch
# import torch.nn.functional as F
# from torch import nn
# import numpy as np
# from util import box_ops
# from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
#                        accuracy, get_world_size, interpolate,
#                        is_dist_avail_and_initialized)
# from function import normal, normal_style
# from function import calc_mean_std
# import scipy.stats as stats
# from models.ViT_helper import DropPath, to_2tuple, trunc_normal_

# class PatchEmbed(nn.Module):
#     """ Image to Patch Embedding
#     """
#     def __init__(self, img_size=256, patch_size=8, in_chans=3, embed_dim=512):
#         super().__init__()
#         img_size = to_2tuple(img_size)
#         patch_size = to_2tuple(patch_size)
#         num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.num_patches = num_patches
        
#         self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

#     def forward(self, x):
#         B, C, H, W = x.shape
#         x = self.proj(x)
#         return x


# class Decoder(nn.Module):
#     def __init__(self):
#         super(Decoder, self).__init__()
#         self.reflection_pad1 = nn.ReflectionPad2d((1, 1, 1, 1))
#         self.conv1 = nn.Conv2d(512, 256, (3, 3))
#         self.relu = nn.ReLU()
#         self.conv2 = nn.Conv2d(256, 256, (3, 3))
#         self.conv3 = nn.Conv2d(256, 256, (3, 3))
#         self.conv4 = nn.Conv2d(256, 128, (3, 3))
#         self.conv5 = nn.Conv2d(128, 128, (3, 3))
#         self.conv6 = nn.Conv2d(128, 64, (3, 3))
#         self.conv7 = nn.Conv2d(64, 64, (3, 3))
#         self.conv8 = nn.Conv2d(64, 3, (3, 3))

#     def forward(self, x):
#         x = self.reflection_pad1(x)
#         x = self.conv1(x)
#         x = self.relu(x)
#         x = F.interpolate(x, scale_factor=2, mode='nearest')
#         x = self.reflection_pad1(x)
#         x = self.conv2(x)
#         x = self.relu(x)
#         x = self.reflection_pad1(x)
#         x = self.conv3(x)
#         x = self.relu(x)
#         x = self.reflection_pad1(x)
#         x = self.conv4(x)
#         x = self.relu(x)
#         x = F.interpolate(x, scale_factor=2, mode='nearest')
#         x = self.reflection_pad1(x)
#         x = self.conv5(x)
#         x = self.relu(x)
#         x = self.reflection_pad1(x)
#         x = self.conv6(x)
#         x = self.relu(x)
#         x = F.interpolate(x, scale_factor=2, mode='nearest')
#         x = self.reflection_pad1(x)
#         x = self.conv7(x)
#         x = self.relu(x)
#         x = self.reflection_pad1(x)
#         x = self.conv8(x)
#         return x


# decoder = Decoder()

# vgg = nn.Sequential(
#     nn.Conv2d(3, 3, (1, 1)),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(3, 64, (3, 3)),
#     nn.ReLU(),  # relu1-1
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(64, 64, (3, 3)),
#     nn.ReLU(),  # relu1-2
#     nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(64, 128, (3, 3)),
#     nn.ReLU(),  # relu2-1
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(128, 128, (3, 3)),
#     nn.ReLU(),  # relu2-2
#     nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(128, 256, (3, 3)),
#     nn.ReLU(),  # relu3-1
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(256, 256, (3, 3)),
#     nn.ReLU(),  # relu3-2
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(256, 256, (3, 3)),
#     nn.ReLU(),  # relu3-3
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(256, 256, (3, 3)),
#     nn.ReLU(),  # relu3-4
#     nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(256, 512, (3, 3)),
#     nn.ReLU(),  # relu4-1, this is the last layer used
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(512, 512, (3, 3)),
#     nn.ReLU(),  # relu4-2
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(512, 512, (3, 3)),
#     nn.ReLU(),  # relu4-3
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(512, 512, (3, 3)),
#     nn.ReLU(),  # relu4-4
#     nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(512, 512, (3, 3)),
#     nn.ReLU(),  # relu5-1
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(512, 512, (3, 3)),
#     nn.ReLU(),  # relu5-2
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(512, 512, (3, 3)),
#     nn.ReLU(),  # relu5-3
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(512, 512, (3, 3)),
#     nn.ReLU()  # relu5-4
# )

# class MLP(nn.Module):
#     """ Very simple multi-layer perceptron (also called FFN)"""

#     def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
#         super().__init__()
#         self.num_layers = num_layers
#         h = [hidden_dim] * (num_layers - 1)
#         self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

#     def forward(self, x):
#         for i, layer in enumerate(self.layers):
#             x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
#         return x

# class StyTrans(nn.Module):
#     """ This is the style transform transformer module """
    
#     def __init__(self, encoder, decoder, PatchEmbed, transformer, args):
#         super().__init__()
#         enc_layers = list(encoder.children())
#         self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
#         self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
#         self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
#         self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
#         self.enc_5 = nn.Sequential(*enc_layers[31:44])  # relu4_1 -> relu5_1
        
#         for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4', 'enc_5']:
#             for param in getattr(self, name).parameters():
#                 param.requires_grad = False

#         self.mse_loss = nn.MSELoss()
#         self.transformer = transformer
#         hidden_dim = transformer.d_model       
#         self.decode = decoder
#         self.embedding = PatchEmbed

#     def encode_with_intermediate(self, input):
#         results = [input]
#         for i in range(5):
#             func = getattr(self, 'enc_{:d}'.format(i + 1))
#             results.append(func(results[-1]))
#         return results[1:]

#     def calc_content_loss(self, input, target):
#         assert (input.size() == target.size())
#         assert (target.requires_grad is False)
#         return self.mse_loss(input, target)

#     def calc_style_loss(self, input, target):
#         assert (input.size() == target.size())
#         assert (target.requires_grad is False)
#         input_mean, input_std = calc_mean_std(input)
#         target_mean, target_std = calc_mean_std(target)
#         return self.mse_loss(input_mean, target_mean) + \
#                self.mse_loss(input_std, target_std)

#     def forward(self, samples_c: NestedTensor, samples_s: NestedTensor):
#         """ The forward expects a NestedTensor, which consists of:
#                - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
#                - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
#         """
#         content_input = samples_c
#         style_input = samples_s
#         if isinstance(samples_c, (list, torch.Tensor)):
#             samples_c = nested_tensor_from_tensor_list(samples_c)   # support different-sized images padding is used for mask [tensor, mask] 
#         if isinstance(samples_s, (list, torch.Tensor)):
#             samples_s = nested_tensor_from_tensor_list(samples_s) 
        
#         # features used to calculate loss 
#         content_feats = self.encode_with_intermediate(samples_c.tensors)
#         style_feats = self.encode_with_intermediate(samples_s.tensors)

#         # Linear projection
#         style = self.embedding(samples_s.tensors)
#         content = self.embedding(samples_c.tensors)
        
#         # positional embedding is calculated in transformer.py
#         pos_s = None
#         pos_c = None

#         mask = None
#         hs = self.transformer(style, mask, content, pos_c, pos_s)   
#         Ics = self.decode(hs)

#         Ics_feats = self.encode_with_intermediate(Ics)
#         loss_c = self.calc_content_loss(normal(Ics_feats[-1]), normal(content_feats[-1])) + self.calc_content_loss(normal(Ics_feats[-2]), normal(content_feats[-2]))
#         # Style loss
#         loss_s = self.calc_style_loss(Ics_feats[0], style_feats[0])
#         for i in range(1, 5):
#             loss_s += self.calc_style_loss(Ics_feats[i], style_feats[i])
        
#         Icc = self.decode(self.transformer(content, mask, content, pos_c, pos_c))
#         Iss = self.decode(self.transformer(style, mask, style, pos_s, pos_s))    

#         # Identity losses lambda 1    
#         loss_lambda1 = self.calc_content_loss(Icc, content_input) + self.calc_content_loss(Iss, style_input)
        
#         # Identity losses lambda 2
#         Icc_feats = self.encode_with_intermediate(Icc)
#         Iss_feats = self.encode_with_intermediate(Iss)
#         loss_lambda2 = self.calc_content_loss(Icc_feats[0], content_feats[0]) + self.calc_content_loss(Iss_feats[0], style_feats[0])
#         for i in range(1, 5):
#             loss_lambda2 += self.calc_content_loss(Icc_feats[i], content_feats[i]) + self.calc_content_loss(Iss_feats[i], style_feats[i])

#         # Please select and comment out one of the following two sentences
#         return Ics,  loss_c, loss_s, loss_lambda1, loss_lambda2   # train
#         # return Ics    # test

