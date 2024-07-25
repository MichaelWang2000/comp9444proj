# train.py
import argparse
import os
import torch
import torch.nn as nn
import torch.utils.data as data
from PIL import Image
from PIL import ImageFile
from tensorboardX import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path
import models.transformer as transformer
import models.StyTR as StyTR
from sampler import InfiniteSamplerWrapper
from torchvision.utils import save_image
from models.gan_model import WGANGPDiscriminator  # 导入判别器
from models.gan_model import Discriminator
import torchvision.utils as vutils
from torchvision.models import vgg19
# 定义其他函数和类（与之前相同）
# ...

class VGGFeatureExtractor(nn.Module):
    def __init__(self, layers):
        super(VGGFeatureExtractor, self).__init__()
        vgg = vgg19(pretrained=True).features
        self.layers = layers
        self.model = nn.Sequential(*[vgg[i] for i in range(max(layers) + 1)])
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = []
        for i, layer in enumerate(self.model):
            x = layer(x)
            if i in self.layers:
                features.append(x)
        return features




def denormalize(tensor):
    return tensor * 0.5 + 0.5


def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        print(self.root)
        self.path = os.listdir(self.root)
        if os.path.isdir(os.path.join(self.root, self.path[0])):
            self.paths = []
            for file_name in os.listdir(self.root):
                for file_name1 in os.listdir(os.path.join(self.root, file_name)):
                    self.paths.append(self.root + "/" + file_name + "/" + file_name1)
        else:
            self.paths = list(Path(self.root).glob('*'))
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'


def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = 2e-4 / (1.0 + args.lr_decay * (iteration_count - 1e4))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr * 0.1 * (1.0 + 3e-4 * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_dir', default='./datasets/train2014', type=str,
                    help='Directory path to a batch of content images')
parser.add_argument('--style_dir', default='./datasets/Images', type=str,  # wikiart dataset crawled from https://www.wikiart.org/
                    help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default='./experiments/vgg_normalised.pth')  # run the train.py, please download the pretrained vgg checkpoint

# training options
parser.add_argument('--save_dir', default='./experiments',
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='./logs',
                    help='Directory to save the log')
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--lr_decay', type=float, default=1e-5)
parser.add_argument('--max_iter', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--style_weight', type=float, default=10.0)
parser.add_argument('--content_weight', type=float, default=7.0)
parser.add_argument('--n_threads', type=int, default=16)
parser.add_argument('--save_model_interval', type=int, default=10000)
parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                    help="Type of positional embedding to use on top of the image features")
parser.add_argument('--hidden_dim', default=512, type=int,
                    help="Size of the embeddings (dimension of the transformer)")
args = parser.parse_args()

USE_CUDA = torch.cuda.is_available()
print(USE_CUDA)
print(torch.cuda.device_count())

device = torch.device("cuda" if USE_CUDA else "cpu")

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

if not os.path.exists(args.log_dir):
    os.mkdir(args.log_dir)
writer = SummaryWriter(log_dir=args.log_dir)

vgg_layers = [2, 7, 12, 21, 30]  # VGG19的特定层
feature_extractor = VGGFeatureExtractor(vgg_layers).to(device)

vgg = StyTR.vgg
vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:44])

decoder = StyTR.decoder
embedding = StyTR.PatchEmbed()

Trans = transformer.Transformer()
with torch.no_grad():
    network = StyTR.StyTrans(vgg, decoder, embedding, Trans, args)
network.train()

network.to(device)
network = nn.DataParallel(network, device_ids=[0, 1])
content_tf = train_transform()
style_tf = train_transform()

content_dataset = FlatFolderDataset(args.content_dir, content_tf)
style_dataset = FlatFolderDataset(args.style_dir, style_tf)

content_iter = iter(data.DataLoader(
    content_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(content_dataset),
    num_workers=args.n_threads))
style_iter = iter(data.DataLoader(
    style_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(style_dataset),
    num_workers=args.n_threads))

# 初始化判别器
params = {
    'nc': 3,      # Number of channels in the input image (e.g., 3 for RGB)
    'ndf': 64     # Number of filters in the first convolutional layer
}
netD = Discriminator(params).to(device)
#netD = WGANGPDiscriminator().to(device)

#############################################
optimizerD = torch.optim.Adam(netD.parameters(), lr=1e-5, betas=(0.5, 0.999))

# 初始化生成器的优化器
optimizerG = torch.optim.Adam([
    {'params': network.module.transformer.parameters()},
    {'params': network.module.decode.parameters()},
    {'params': network.module.embedding.parameters()},
], lr=args.lr)

def gradient_penalty(D, real_samples, fake_samples, device):
    alpha = torch.randn(real_samples.size(0), 1, 1, 1).to(device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones(d_interpolates.size()).to(device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty




criterion = nn.BCELoss()
mse_loss = nn.MSELoss()

if not os.path.exists(args.save_dir + "/test"):
    os.makedirs(args.save_dir + "/test")


# for i in tqdm(range(args.max_iter)):

#     if i < 1e4:
#         warmup_learning_rate(optimizerG, iteration_count=i)
#     else:
#         adjust_learning_rate(optimizerG, iteration_count=i)

#     content_images = next(content_iter).to(device)
#     style_images = next(style_iter).to(device)

#     # 训练判别器
#     netD.zero_grad()
#     real_images = content_images
#     real_labels = torch.full((real_images.size(0),), 1, device=device, dtype=torch.float)  # 确保 real_labels 是 float 类型
#     output = netD(real_images).view(-1)
#     errD_real = -torch.mean(output)  # WGAN的真实样本损失

#     fake_images, _, _, _, _ = network(content_images, style_images)
#     output = netD(fake_images.detach()).view(-1)
#     errD_fake = torch.mean(output)  # WGAN的生成样本损失

#     # print(fake_images.shape)
#     # print(fake_images.min(), fake_images.max())

#     # 梯度惩罚
#     gp = gradient_penalty(netD, real_images, fake_images, device)
#     #errD = errD_real + errD_fake + 10 * gp  # WGAN-GP总损失
#     errD = errD_real + errD_fake + 1 * gp
#     errD.backward(retain_graph=True)  # 保留计算图
#     optimizerD.step()

#     # 训练生成器
#     network.zero_grad()
#     output = netD(fake_images).view(-1)
#     errG = -torch.mean(output)  # WGAN的生成器损失
#     errG.backward(retain_graph=True)  # 保留计算图
#     optimizerG.step()
    
#     if i % 100 == 0:
#         output_name = '{:s}/test/{:s}{:s}'.format(
#             args.save_dir, str(i), ".jpg"
#         )
#         out = torch.cat((content_images, fake_images), 0)
#         out = torch.cat((style_images, out), 0)
#         save_image(out, output_name)

#         fake_images_denorm = denormalize(fake_images)
#         vutils.save_image(fake_images_denorm, '{:s}/test/fake_images_{:d}.png'.format(args.save_dir, i), normalize=False)


#     print(f"[{i}/{args.max_iter}] Loss_D: {errD.item()}, Loss_G: {errG.item()}")

#     writer.add_scalar('Loss_D', errD.item(), i + 1)
#     writer.add_scalar('Loss_G', errG.item(), i + 1)

#     if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
#         state_dict = network.module.transformer.state_dict()
#         for key in state_dict.keys():
#             state_dict[key] = state_dict[key].to(torch.device('cpu'))
#         torch.save(state_dict,
#                    '{:s}/transformer_iter_{:d}.pth'.format(args.save_dir,
#                                                            i + 1))

#         state_dict = network.module.decode.state_dict()
#         for key in state_dict.keys():
#             state_dict[key] = state_dict[key].to(torch.device('cpu'))
#         torch.save(state_dict,
#                    '{:s}/decoder_iter_{:d}.pth'.format(args.save_dir,
#                                                        i + 1))
#         state_dict = network.module.embedding.state_dict()
#         for key in state_dict.keys():
#             state_dict[key] = state_dict[key].to(torch.device('cpu'))
#         torch.save(state_dict,
#                    '{:s}/embedding_iter_{:d}.pth'.format(args.save_dir,
#                                                          i + 1))

# writer.close()












    
for i in tqdm(range(args.max_iter)):

    if i < 1e4:
        warmup_learning_rate(optimizerG, iteration_count=i)
    else:
        adjust_learning_rate(optimizerG, iteration_count=i)

    content_images = next(content_iter).to(device)
    style_images = next(style_iter).to(device)

    # 训练判别器
    netD.zero_grad()
    # #real_images = style_images
    # real_images = torch.cat((content_images, style_images), dim=1)
    # real_labels = torch.full((real_images.size(0),), 1, device=device, dtype=torch.float)  # 确保 real_labels 是 float 类型
    # output = netD(real_images).view(-1)
    # real_labels = torch.full(output.size(), 1, device=device, dtype=torch.float)  # 调整 real_labels 的大小并转换为 float 类型
    # errD_real = criterion(output, real_labels)
    # errD_real.backward()

    real_labels = torch.full((content_images.size(0),), 1, device=device, dtype=torch.float)
    output = netD(content_images, style_images).view(-1)
    real_labels = torch.full(output.size(), 1, device=device, dtype=torch.float)
    errD_real = criterion(output, real_labels)
    errD_real.backward()

    

    

    #fake_images, _, _, _, _ = network(content_images, style_images)
    fake_images, loss_c, loss_s,l_identity1, l_identity2 = network(content_images, style_images)
    #output = netD(fake_images.detach()).view(-1)
    output = netD(fake_images.detach(), style_images).view(-1)
    fake_labels = torch.full(output.size(), 0, device=device, dtype=torch.float)  # 调整 fake_labels 的大小并转换为 float 类型
    errD_fake = criterion(output, fake_labels)
    errD_fake.backward()
    optimizerD.step()

    # 训练生成器
    network.zero_grad()
    #output = netD(fake_images).view(-1)
    output = netD(fake_images, style_images).view(-1)
    real_labels = torch.full(output.size(), 1, device=device, dtype=torch.float)  # 重新调整 real_labels 的大小并转换为 float 类型
    errG = criterion(output, real_labels)  # 生成器希望判别器认为这些是real images
    #errG.backward()
    #optimizerG.step()

    real_features = feature_extractor(style_images)
    fake_features = feature_extractor(fake_images)
    perceptual_loss = sum([mse_loss(r, f) for r, f in zip(real_features, fake_features)])

    # 总损失
    total_loss = errG + perceptual_loss * 0.1 + loss_c * 0.5 + loss_s * 1.0
    total_loss.backward()
    optimizerG.step()

    if i % 100 == 0:
        output_name = '{:s}/test/{:s}{:s}'.format(
            args.save_dir, str(i), ".jpg"
        )
        out = torch.cat((content_images, fake_images), 0)
        out = torch.cat((style_images, out), 0)
        save_image(out, output_name)
        fake_images_denorm = denormalize(fake_images)
        vutils.save_image(fake_images_denorm, '{:s}/test/fake_images_{:d}.png'.format(args.save_dir, i), normalize=False)

    print(f"[{i}/{args.max_iter}] Loss_D: {errD_real.item() + errD_fake.item()}, Loss_G: {errG.item()}")

    loss_c = args.content_weight * loss_c
    loss_s = args.style_weight * loss_s
    loss = loss_c + loss_s + (l_identity1 * 70) + (l_identity2 * 1) 
  
    print(loss.sum().cpu().detach().numpy(),"-content:",loss_c.sum().cpu().detach().numpy(),"-style:",loss_s.sum().cpu().detach().numpy()
              ,"-l1:",l_identity1.sum().cpu().detach().numpy(),"-l2:",l_identity2.sum().cpu().detach().numpy()
              )

    writer.add_scalar('Loss_D', errD_real.item() + errD_fake.item(), i + 1)
    writer.add_scalar('Loss_G', errG.item(), i + 1)
    
    writer.add_scalar('loss_content', loss_c.sum().item(), i + 1)
    writer.add_scalar('loss_style', loss_s.sum().item(), i + 1)
    writer.add_scalar('loss_identity1', l_identity1.sum().item(), i + 1)
    writer.add_scalar('loss_identity2', l_identity2.sum().item(), i + 1)
    writer.add_scalar('total_loss', loss.sum().item(), i + 1)
    

    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
        state_dict = network.module.transformer.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,
                   '{:s}/transformer_iter_{:d}.pth'.format(args.save_dir,
                                                           i + 1))

        state_dict = network.module.decode.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,
                   '{:s}/decoder_iter_{:d}.pth'.format(args.save_dir,
                                                       i + 1))
        state_dict = network.module.embedding.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,
                   '{:s}/embedding_iter_{:d}.pth'.format(args.save_dir,
                                                         i + 1))

writer.close()



















# # train.py
# import argparse
# import os
# import torch
# import torch.nn as nn
# import torch.utils.data as data
# from PIL import Image
# from PIL import ImageFile
# from tensorboardX import SummaryWriter
# from torchvision import transforms
# from tqdm import tqdm
# from pathlib import Path
# import models.transformer as transformer
# import models.StyTR as StyTR
# from sampler import InfiniteSamplerWrapper
# from torchvision.utils import save_image
# from models.gan_model import Discriminator  # 导入判别器

# # 定义其他函数和类（与之前相同）
# # ...

# def train_transform():
#     transform_list = [
#         transforms.Resize(size=(512, 512)),
#         transforms.RandomCrop(256),
#         transforms.ToTensor()
#     ]
#     return transforms.Compose(transform_list)


# class FlatFolderDataset(data.Dataset):
#     def __init__(self, root, transform):
#         super(FlatFolderDataset, self).__init__()
#         self.root = root
#         print(self.root)
#         self.path = os.listdir(self.root)
#         if os.path.isdir(os.path.join(self.root, self.path[0])):
#             self.paths = []
#             for file_name in os.listdir(self.root):
#                 for file_name1 in os.listdir(os.path.join(self.root, file_name)):
#                     self.paths.append(self.root + "/" + file_name + "/" + file_name1)
#         else:
#             self.paths = list(Path(self.root).glob('*'))
#         self.transform = transform

#     def __getitem__(self, index):
#         path = self.paths[index]
#         img = Image.open(str(path)).convert('RGB')
#         img = self.transform(img)
#         return img

#     def __len__(self):
#         return len(self.paths)

#     def name(self):
#         return 'FlatFolderDataset'


# def adjust_learning_rate(optimizer, iteration_count):
#     """Imitating the original implementation"""
#     lr = 2e-4 / (1.0 + args.lr_decay * (iteration_count - 1e4))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


# def warmup_learning_rate(optimizer, iteration_count):
#     """Imitating the original implementation"""
#     lr = args.lr * 0.1 * (1.0 + 3e-4 * iteration_count)
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


# parser = argparse.ArgumentParser()
# # Basic options
# parser.add_argument('--content_dir', default='./datasets/train2014', type=str,
#                     help='Directory path to a batch of content images')
# parser.add_argument('--style_dir', default='./datasets/Images', type=str,  # wikiart dataset crawled from https://www.wikiart.org/
#                     help='Directory path to a batch of style images')
# parser.add_argument('--vgg', type=str, default='./experiments/vgg_normalised.pth')  # run the train.py, please download the pretrained vgg checkpoint

# # training options
# parser.add_argument('--save_dir', default='./experiments',
#                     help='Directory to save the model')
# parser.add_argument('--log_dir', default='./logs',
#                     help='Directory to save the log')
# parser.add_argument('--lr', type=float, default=5e-4)
# parser.add_argument('--lr_decay', type=float, default=1e-5)
# parser.add_argument('--max_iter', type=int, default=1000)
# parser.add_argument('--batch_size', type=int, default=8)
# parser.add_argument('--style_weight', type=float, default=10.0)
# parser.add_argument('--content_weight', type=float, default=7.0)
# parser.add_argument('--n_threads', type=int, default=16)
# parser.add_argument('--save_model_interval', type=int, default=10000)
# parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
#                     help="Type of positional embedding to use on top of the image features")
# parser.add_argument('--hidden_dim', default=512, type=int,
#                     help="Size of the embeddings (dimension of the transformer)")
# args = parser.parse_args()

# USE_CUDA = torch.cuda.is_available()
# print(USE_CUDA)
# print(torch.cuda.device_count())

# device = torch.device("cuda" if USE_CUDA else "cpu")

# if not os.path.exists(args.save_dir):
#     os.makedirs(args.save_dir)

# if not os.path.exists(args.log_dir):
#     os.mkdir(args.log_dir)
# writer = SummaryWriter(log_dir=args.log_dir)

# vgg = StyTR.vgg
# vgg.load_state_dict(torch.load(args.vgg))
# vgg = nn.Sequential(*list(vgg.children())[:44])

# decoder = StyTR.decoder
# embedding = StyTR.PatchEmbed()

# Trans = transformer.Transformer()
# with torch.no_grad():
#     network = StyTR.StyTrans(vgg, decoder, embedding, Trans, args)
# network.train()

# network.to(device)
# network = nn.DataParallel(network, device_ids=[0, 1])
# content_tf = train_transform()
# style_tf = train_transform()

# content_dataset = FlatFolderDataset(args.content_dir, content_tf)
# style_dataset = FlatFolderDataset(args.style_dir, style_tf)

# content_iter = iter(data.DataLoader(
#     content_dataset, batch_size=args.batch_size,
#     sampler=InfiniteSamplerWrapper(content_dataset),
#     num_workers=args.n_threads))
# style_iter = iter(data.DataLoader(
#     style_dataset, batch_size=args.batch_size,
#     sampler=InfiniteSamplerWrapper(style_dataset),
#     num_workers=args.n_threads))

# # 初始化判别器
# netD = Discriminator().to(device)

# #############################################
# optimizerD = torch.optim.Adam(netD.parameters(), lr=5e-5, betas=(0.5, 0.999))

# # 初始化生成器的优化器
# optimizerG = torch.optim.Adam([
#     {'params': network.module.transformer.parameters()},
#     {'params': network.module.decode.parameters()},
#     {'params': network.module.embedding.parameters()},
# ], lr=args.lr)

# criterion = nn.BCELoss()

# if not os.path.exists(args.save_dir + "/test"):
#     os.makedirs(args.save_dir + "/test")

# for i in tqdm(range(args.max_iter)):

#     if i < 1e4:
#         warmup_learning_rate(optimizerG, iteration_count=i)
#     else:
#         adjust_learning_rate(optimizerG, iteration_count=i)

#     content_images = next(content_iter).to(device)
#     style_images = next(style_iter).to(device)

#     # 训练判别器
#     netD.zero_grad()
#     real_images = content_images
#     real_labels = torch.full((real_images.size(0),), 1, device=device, dtype=torch.float)  # 确保 real_labels 是 float 类型
#     output = netD(real_images).view(-1)
#     real_labels = torch.full(output.size(), 1, device=device, dtype=torch.float)  # 调整 real_labels 的大小并转换为 float 类型
#     errD_real = criterion(output, real_labels)
#     errD_real.backward()

#     fake_images, _, _, _, _ = network(content_images, style_images)
#     output = netD(fake_images.detach()).view(-1)
#     fake_labels = torch.full(output.size(), 0, device=device, dtype=torch.float)  # 调整 fake_labels 的大小并转换为 float 类型
#     errD_fake = criterion(output, fake_labels)
#     errD_fake.backward()
#     optimizerD.step()

#     # 训练生成器
#     network.zero_grad()
#     output = netD(fake_images).view(-1)
#     real_labels = torch.full(output.size(), 1, device=device, dtype=torch.float)  # 重新调整 real_labels 的大小并转换为 float 类型
#     errG = criterion(output, real_labels)  # 生成器希望判别器认为这些是real images
#     errG.backward()
#     optimizerG.step()

#     if i % 100 == 0:
#         output_name = '{:s}/test/{:s}{:s}'.format(
#             args.save_dir, str(i), ".jpg"
#         )
#         out = torch.cat((content_images, fake_images), 0)
#         out = torch.cat((style_images, out), 0)
#         save_image(out, output_name)

#     print(f"[{i}/{args.max_iter}] Loss_D: {errD_real.item() + errD_fake.item()}, Loss_G: {errG.item()}")

#     writer.add_scalar('Loss_D', errD_real.item() + errD_fake.item(), i + 1)
#     writer.add_scalar('Loss_G', errG.item(), i + 1)

#     if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
#         state_dict = network.module.transformer.state_dict()
#         for key in state_dict.keys():
#             state_dict[key] = state_dict[key].to(torch.device('cpu'))
#         torch.save(state_dict,
#                    '{:s}/transformer_iter_{:d}.pth'.format(args.save_dir,
#                                                            i + 1))

#         state_dict = network.module.decode.state_dict()
#         for key in state_dict.keys():
#             state_dict[key] = state_dict[key].to(torch.device('cpu'))
#         torch.save(state_dict,
#                    '{:s}/decoder_iter_{:d}.pth'.format(args.save_dir,
#                                                        i + 1))
#         state_dict = network.module.embedding.state_dict()
#         for key in state_dict.keys():
#             state_dict[key] = state_dict[key].to(torch.device('cpu'))
#         torch.save(state_dict,
#                    '{:s}/embedding_iter_{:d}.pth'.format(args.save_dir,
#                                                          i + 1))

# writer.close()