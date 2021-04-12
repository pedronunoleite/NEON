import os
import torch
import torch.nn as nn
import torchvision.models
import collections
import math
import torch.nn.functional as F
import kornia

class Identity(nn.Module):
    # a dummy identity module
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class Unpool(nn.Module):
    # Unpool: 2*2 unpooling with zero padding
    def __init__(self, stride=2):
        super(Unpool, self).__init__()

        self.stride = stride

        # create kernel [1, 0; 0, 0]
        self.mask = torch.zeros(1, 1, stride, stride)
        self.mask[:,:,0,0] = 1

    def forward(self, x):
        assert x.dim() == 4
        num_channels = x.size(1)
        return F.conv_transpose2d(x,
            self.mask.detach().type_as(x).expand(num_channels, 1, -1, -1),
            stride=self.stride, groups=num_channels)

def weights_init(m):
    # Initialize kernel weights with Gaussian distributions
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

def conv(in_channels, out_channels, kernel_size):
    padding = (kernel_size-1) // 2
    assert 2*padding == kernel_size-1, "parameters incorrect. kernel={}, padding={}".format(kernel_size, padding)
    return nn.Sequential(
          nn.Conv2d(in_channels,out_channels,kernel_size,stride=1,padding=padding,bias=False),
          nn.BatchNorm2d(out_channels),
          nn.ReLU(inplace=True),
        )

def depthwise(in_channels, kernel_size):
    padding = (kernel_size-1) // 2
    assert 2*padding == kernel_size-1, "parameters incorrect. kernel={}, padding={}".format(kernel_size, padding)
    return nn.Sequential(
          nn.Conv2d(in_channels,in_channels,kernel_size,stride=1,padding=padding,bias=False,groups=in_channels),
          nn.BatchNorm2d(in_channels),
          nn.ReLU(inplace=True),
        )

def pointwise(in_channels, out_channels):
    return nn.Sequential(
          nn.Conv2d(in_channels,out_channels,1,1,0,bias=False),
          nn.BatchNorm2d(out_channels),
          nn.ReLU(inplace=True),
        )

def convt(in_channels, out_channels, kernel_size):
    stride = 2
    padding = (kernel_size - 1) // 2
    output_padding = kernel_size % 2
    assert -2 - 2*padding + kernel_size + output_padding == 0, "deconv parameters incorrect"
    return nn.Sequential(
            nn.ConvTranspose2d(in_channels,out_channels,kernel_size,
                stride,padding,output_padding,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

def convt_dw(channels, kernel_size):
    stride = 2
    padding = (kernel_size - 1) // 2
    output_padding = kernel_size % 2
    assert -2 - 2*padding + kernel_size + output_padding == 0, "deconv parameters incorrect"
    return nn.Sequential(
            nn.ConvTranspose2d(channels,channels,kernel_size,
                stride,padding,output_padding,bias=False,groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

def upconv(in_channels, out_channels):
    return nn.Sequential(
        Unpool(2),
        nn.Conv2d(in_channels,out_channels,kernel_size=5,stride=1,padding=2,bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )

class upproj(nn.Module):
    # UpProj module has two branches, with a Unpool at the start and a ReLu at the end
    #   upper branch: 5*5 conv -> batchnorm -> ReLU -> 3*3 conv -> batchnorm
    #   bottom branch: 5*5 conv -> batchnorm

    def __init__(self, in_channels, out_channels):
        super(upproj, self).__init__()
        self.unpool = Unpool(2)
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=5,stride=1,padding=2,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=5,stride=1,padding=2,bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = self.unpool(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        return F.relu(x1 + x2)

class Decoder(nn.Module):
    names = ['deconv{}{}'.format(i,dw) for i in range(3,10,2) for dw in ['', 'dw']]
    names.append("upconv")
    names.append("upproj")
    for i in range(3,10,2):
        for dw in ['', 'dw']:
            names.append("nnconv{}{}".format(i, dw))
            names.append("blconv{}{}".format(i, dw))
            names.append("shuffle{}{}".format(i, dw))

class DeConv(nn.Module):

    def __init__(self, kernel_size, dw):
        super(DeConv, self).__init__()
        if dw:
            self.convt1 = nn.Sequential(
                convt_dw(1024, kernel_size),
                pointwise(1024, 512))
            self.convt2 = nn.Sequential(
                convt_dw(512, kernel_size),
                pointwise(512, 256))
            self.convt3 = nn.Sequential(
                convt_dw(256, kernel_size),
                pointwise(256, 128))
            self.convt4 = nn.Sequential(
                convt_dw(128, kernel_size),
                pointwise(128, 64))
            self.convt5 = nn.Sequential(
                convt_dw(64, kernel_size),
                pointwise(64, 32))
        else:
            self.convt1 = convt(1024, 512, kernel_size)
            self.convt2 = convt(512, 256, kernel_size)
            self.convt3 = convt(256, 128, kernel_size)
            self.convt4 = convt(128, 64, kernel_size)
            self.convt5 = convt(64, 32, kernel_size)
        self.convf = pointwise(32, 1)

    def forward(self, x):
        x = self.convt1(x)
        x = self.convt2(x)
        x = self.convt3(x)
        x = self.convt4(x)
        x = self.convt5(x)
        x = self.convf(x)
        return x


class UpConv(nn.Module):

    def __init__(self):
        super(UpConv, self).__init__()
        self.upconv1 = upconv(1024, 512)
        self.upconv2 = upconv(512, 256)
        self.upconv3 = upconv(256, 128)
        self.upconv4 = upconv(128, 64)
        self.upconv5 = upconv(64, 32)
        self.convf = pointwise(32, 1)

    def forward(self, x):
        x = self.upconv1(x)
        x = self.upconv2(x)
        x = self.upconv3(x)
        x = self.upconv4(x)
        x = self.upconv5(x)
        x = self.convf(x)
        return x

class UpProj(nn.Module):
    # UpProj decoder consists of 4 upproj modules with decreasing number of channels and increasing feature map size

    def __init__(self):
        super(UpProj, self).__init__()
        self.upproj1 = upproj(1024, 512)
        self.upproj2 = upproj(512, 256)
        self.upproj3 = upproj(256, 128)
        self.upproj4 = upproj(128, 64)
        self.upproj5 = upproj(64, 32)
        self.convf = pointwise(32, 1)

    def forward(self, x):
        x = self.upproj1(x)
        x = self.upproj2(x)
        x = self.upproj3(x)
        x = self.upproj4(x)
        x = self.upproj5(x)
        x = self.convf(x)
        return x

class NNConv(nn.Module):

    def __init__(self, kernel_size, dw):
        super(NNConv, self).__init__()
        if dw:
            self.conv1 = nn.Sequential(
                depthwise(1024, kernel_size),
                pointwise(1024, 512))
            self.conv2 = nn.Sequential(
                depthwise(512, kernel_size),
                pointwise(512, 256))
            self.conv3 = nn.Sequential(
                depthwise(256, kernel_size),
                pointwise(256, 128))
            self.conv4 = nn.Sequential(
                depthwise(128, kernel_size),
                pointwise(128, 64))
            self.conv5 = nn.Sequential(
                depthwise(64, kernel_size),
                pointwise(64, 32))
            self.conv6 = pointwise(32, 1)
        else:
            self.conv1 = conv(1024, 512, kernel_size)
            self.conv2 = conv(512, 256, kernel_size)
            self.conv3 = conv(256, 128, kernel_size)
            self.conv4 = conv(128, 64, kernel_size)
            self.conv5 = conv(64, 32, kernel_size)
            self.conv6 = pointwise(32, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')

        x = self.conv2(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')

        x = self.conv3(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')

        x = self.conv4(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')

        x = self.conv5(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')

        x = self.conv6(x)
        return x

class BLConv(NNConv):

    def __init__(self, kernel_size, dw):
        super(BLConv, self).__init__(kernel_size, dw)

    def forward(self, x):
        x = self.conv1(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        x = self.conv2(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        x = self.conv3(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        x = self.conv4(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        x = self.conv5(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        x = self.conv6(x)
        return x

class ShuffleConv(nn.Module):

    def __init__(self, kernel_size, dw):
        super(ShuffleConv, self).__init__()
        if dw:
            self.conv1 = nn.Sequential(
                depthwise(256, kernel_size),
                pointwise(256, 256))
            self.conv2 = nn.Sequential(
                depthwise(64, kernel_size),
                pointwise(64, 64))
            self.conv3 = nn.Sequential(
                depthwise(16, kernel_size),
                pointwise(16, 16))
            self.conv4 = nn.Sequential(
                depthwise(4, kernel_size),
                pointwise(4, 4))
        else:
            self.conv1 = conv(256, 256, kernel_size)
            self.conv2 = conv(64, 64, kernel_size)
            self.conv3 = conv(16, 16, kernel_size)
            self.conv4 = conv(4, 4, kernel_size)

    def forward(self, x):
        x = F.pixel_shuffle(x, 2)
        x = self.conv1(x)

        x = F.pixel_shuffle(x, 2)
        x = self.conv2(x)

        x = F.pixel_shuffle(x, 2)
        x = self.conv3(x)

        x = F.pixel_shuffle(x, 2)
        x = self.conv4(x)

        x = F.pixel_shuffle(x, 2)
        return x

def choose_decoder(decoder):
    depthwise = ('dw' in decoder)
    if decoder[:6] == 'deconv':
        assert len(decoder)==7 or (len(decoder)==9 and 'dw' in decoder)
        kernel_size = int(decoder[6])
        model = DeConv(kernel_size, depthwise)
    elif decoder == "upproj":
        model = UpProj()
    elif decoder == "upconv":
        model = UpConv()
    elif decoder[:7] == 'shuffle':
        assert len(decoder)==8 or (len(decoder)==10 and 'dw' in decoder)
        kernel_size = int(decoder[7])
        model = ShuffleConv(kernel_size, depthwise)
    elif decoder[:6] == 'nnconv':
        assert len(decoder)==7 or (len(decoder)==9 and 'dw' in decoder)
        kernel_size = int(decoder[6])
        model = NNConv(kernel_size, depthwise)
    elif decoder[:6] == 'blconv':
        assert len(decoder)==7 or (len(decoder)==9 and 'dw' in decoder)
        kernel_size = int(decoder[6])
        model = BLConv(kernel_size, depthwise)
    else:
        assert False, "invalid option for decoder: {}".format(decoder)
    model.apply(weights_init)
    return model


class ResNet(nn.Module):
    def __init__(self, layers, decoder, output_size, in_channels=3, pretrained=True):

        if layers not in [18, 34, 50, 101, 152]:
            raise RuntimeError('Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))
        
        super(ResNet, self).__init__()
        self.output_size = output_size
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(layers)](pretrained=pretrained)
        if not pretrained:
            pretrained_model.apply(weights_init)
        
        if in_channels == 3:
            self.conv1 = pretrained_model._modules['conv1']
            self.bn1 = pretrained_model._modules['bn1']
        else:
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            weights_init(self.conv1)
            weights_init(self.bn1)
        
        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']
        self.layer1 = pretrained_model._modules['layer1']
        self.layer2 = pretrained_model._modules['layer2']
        self.layer3 = pretrained_model._modules['layer3']
        self.layer4 = pretrained_model._modules['layer4']

        # clear memory
        del pretrained_model

        # define number of intermediate channels
        if layers <= 34:
            num_channels = 512
        elif layers >= 50:
            num_channels = 2048
        self.conv2 = nn.Conv2d(num_channels, 1024, 1)
        weights_init(self.conv2)
        self.decoder = choose_decoder(decoder)

    def forward(self, x):
        # resnet
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv2(x)

        # decoder
        x = self.decoder(x)

        return x

class MobileNet(nn.Module):
    def __init__(self, decoder, output_size, in_channels=3, pretrained=True):

        super(MobileNet, self).__init__()
        self.output_size = output_size
        mobilenet = imagenet.mobilenet.MobileNet()
        if pretrained:
            pretrained_path = os.path.join('imagenet', 'results', 'imagenet.arch=mobilenet.lr=0.1.bs=256', 'model_best.pth.tar')
            checkpoint = torch.load(pretrained_path)
            state_dict = checkpoint['state_dict']

            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            mobilenet.load_state_dict(new_state_dict)
        else:
            mobilenet.apply(weights_init)

        if in_channels == 3:
            self.mobilenet = nn.Sequential(*(mobilenet.model[i] for i in range(14)))
        else:
            def conv_bn(inp, oup, stride):
                return nn.Sequential(
                    nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                    nn.BatchNorm2d(oup),
                    nn.ReLU6(inplace=True)
                )

            self.mobilenet = nn.Sequential(
                conv_bn(in_channels,  32, 2),
                *(mobilenet.model[i] for i in range(1,14))
                )

        self.decoder = choose_decoder(decoder)

    def forward(self, x):
        x = self.mobilenet(x)
        x = self.decoder(x)
        return x

class ResNetSkipAdd(nn.Module):
    def __init__(self, layers, output_size, in_channels=3, pretrained=True):

        self.dw = True

        if layers not in [18, 34, 50, 101, 152]:
            raise RuntimeError('Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))
        
        super(ResNetSkipAdd, self).__init__()
        self.output_size = output_size
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(layers)](pretrained=pretrained)
        if not pretrained:
            pretrained_model.apply(weights_init)
        
        if in_channels == 3:
            self.conv1 = pretrained_model._modules['conv1']
            self.bn1 = pretrained_model._modules['bn1']
        else:
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            weights_init(self.conv1)
            weights_init(self.bn1)
        
        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']
        self.layer1 = pretrained_model._modules['layer1']
        self.layer2 = pretrained_model._modules['layer2']
        self.layer3 = pretrained_model._modules['layer3']
        self.layer4 = pretrained_model._modules['layer4']

        # clear memory
        del pretrained_model

        # define number of intermediate channels
        if layers <= 34:
            num_channels = 512
        elif layers >= 50:
            num_channels = 2048
        self.conv2 = nn.Conv2d(num_channels, 1024, 1)
        weights_init(self.conv2)
        

        kernel_size = 5
        if self.dw :
            self.decode_conv1 = nn.Sequential(
                depthwise(1024, kernel_size),
                pointwise(1024, 512))
            self.decode_conv2 = nn.Sequential(
                depthwise(512, kernel_size),
                pointwise(512, 256))
            self.decode_conv3 = nn.Sequential(
                depthwise(256, kernel_size),
                pointwise(256, 128))
            self.decode_conv4 = nn.Sequential(
                depthwise(128, kernel_size),
                pointwise(128, 64))
            self.decode_conv5 = nn.Sequential(
                depthwise(64, kernel_size),
                pointwise(64, 32))
            self.decode_conv6 = pointwise(32, 1)
        else:
            self.decode_conv1 = conv(1024, 512, kernel_size)
            self.decode_conv2 = conv(512, 256, kernel_size)
            self.decode_conv3 = conv(256, 128, kernel_size)
            self.decode_conv4 = conv(128, 64, kernel_size)
            self.decode_conv5 = conv(64, 32, kernel_size)
            self.decode_conv6 = pointwise(32, 1)

        weights_init(self.decode_conv1)
        weights_init(self.decode_conv2)
        weights_init(self.decode_conv3)
        weights_init(self.decode_conv4)
        weights_init(self.decode_conv5)
        weights_init(self.decode_conv6)

    def forward(self, x):
        # resnet
        x = self.conv1(x)
        #print(x.size())
        x = self.bn1(x)
        #print(x.size())
        x1 = self.relu(x)
        #print("x1", x1.size())
        x2 = self.maxpool(x1)
        #print("x2", x2.size())
        x3 = self.layer1(x2)
        #print("x3", x3.size())
        x4 = self.layer2(x3)
        #print("x4", x4.size())
        x5 = self.layer3(x4)
        #print("x5", x5.size())
        x6 = self.layer4(x5)
        #print("x6", x6.size())
        x7 = self.conv2(x6)
        #print("x7", x7.size())
        # decoder
        y10 = self.decode_conv1(x7)
        #print("y10", y10.size())
        y9 = F.interpolate(y10 + x6, scale_factor=2, mode='bilinear')
        #print("y9", y9.size())
        y8 = self.decode_conv2(y9)
        #print("y8", y8.size())
        y7 = F.interpolate(y8 + x5, scale_factor=2, mode='bilinear')
        #print("y7", y7.size())
        y6 = self.decode_conv3(y7)
        #print("y6", y6.size())
        y5 = F.interpolate(y6 + x4, scale_factor=2, mode='bilinear')
        #print("y5", y5.size())
        y4 = self.decode_conv4(y5)
        #print("y4", y4.size())
        y3 = F.interpolate(y4 + x3, scale_factor=2, mode='bilinear')
        #print("y3", y3.size())
        y2 = self.decode_conv5(y3 + x1)
        #print("y2", y2.size())
        y1 = F.interpolate(y2, scale_factor=2, mode='bilinear')
        #print("y1", y1.size())
        y = self.decode_conv6(y1)
        return y

class ResNetSkipConcat(nn.Module):
    def __init__(self, layers, output_size, in_channels=3, pretrained=True):

        if layers not in [18, 34, 50, 101, 152]:
            raise RuntimeError('Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))
        
        super(ResNetSkipConcat, self).__init__()
        self.output_size = output_size
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(layers)](pretrained=pretrained)
        if not pretrained:
            pretrained_model.apply(weights_init)
        
        if in_channels == 3:
            self.conv1 = pretrained_model._modules['conv1']
            self.bn1 = pretrained_model._modules['bn1']
        else:
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            weights_init(self.conv1)
            weights_init(self.bn1)
        
        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']
        self.layer1 = pretrained_model._modules['layer1']
        self.layer2 = pretrained_model._modules['layer2']
        self.layer3 = pretrained_model._modules['layer3']
        self.layer4 = pretrained_model._modules['layer4']

        # clear memory
        del pretrained_model

        # define number of intermediate channels
        if layers <= 34:
            num_channels = 512
        elif layers >= 50:
            num_channels = 2048
        self.conv2 = nn.Conv2d(num_channels, 1024, 1)
        weights_init(self.conv2)
        
        kernel_size = 5
        self.decode_conv1 = conv(1024, 512, kernel_size)
        self.decode_conv2 = conv(768, 256, kernel_size)
        self.decode_conv3 = conv(384, 128, kernel_size)
        self.decode_conv4 = conv(192, 64, kernel_size)
        self.decode_conv5 = conv(128, 32, kernel_size)
        self.decode_conv6 = pointwise(32, 1)
        weights_init(self.decode_conv1)
        weights_init(self.decode_conv2)
        weights_init(self.decode_conv3)
        weights_init(self.decode_conv4)
        weights_init(self.decode_conv5)
        weights_init(self.decode_conv6)

    def forward(self, x):
        # resnet
        x = self.conv1(x)
        x = self.bn1(x)
        x1 = self.relu(x)
        # print("x1", x1.size())
        x2 = self.maxpool(x1)
        # print("x2", x2.size())
        x3 = self.layer1(x2)
        # print("x3", x3.size())
        x4 = self.layer2(x3)
        # print("x4", x4.size())
        x5 = self.layer3(x4)
        # print("x5", x5.size())
        x6 = self.layer4(x5)
        # print("x6", x6.size())
        x7 = self.conv2(x6)

        # decoder
        y10 = self.decode_conv1(x7)
        # print("y10", y10.size())
        y9 = F.interpolate(y10, scale_factor=2, mode='nearest')
        # print("y9", y9.size())
        y8 = self.decode_conv2(torch.cat((y9, x5), 1))
        # print("y8", y8.size())
        y7 = F.interpolate(y8, scale_factor=2, mode='nearest')
        # print("y7", y7.size())
        y6 = self.decode_conv3(torch.cat((y7, x4), 1))
        # print("y6", y6.size())
        y5 = F.interpolate(y6, scale_factor=2, mode='nearest')
        # print("y5", y5.size())
        y4 = self.decode_conv4(torch.cat((y5, x3), 1))
        # print("y4", y4.size())
        y3 = F.interpolate(y4, scale_factor=2, mode='nearest')
        # print("y3", y3.size())
        y2 = self.decode_conv5(torch.cat((y3, x1), 1))
        # print("y2", y2.size())
        y1 = F.interpolate(y2, scale_factor=2, mode='nearest')
        # print("y1", y1.size())
        y = self.decode_conv6(y1)

        return y

class MobileNetSkipAdd(nn.Module):
    def __init__(self, output_size, pretrained=True):

        super(MobileNetSkipAdd, self).__init__()
        self.output_size = output_size
        mobilenet = imagenet.mobilenet.MobileNet()
        if pretrained:
            pretrained_path = os.path.join('imagenet', 'results', 'imagenet.arch=mobilenet.lr=0.1.bs=256', 'model_best.pth.tar')
            checkpoint = torch.load(pretrained_path)
            state_dict = checkpoint['state_dict']

            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            mobilenet.load_state_dict(new_state_dict)
        else:
            mobilenet.apply(weights_init)

        for i in range(14):
            setattr( self, 'conv{}'.format(i), mobilenet.model[i])

        kernel_size = 5
        # self.decode_conv1 = conv(1024, 512, kernel_size)
        # self.decode_conv2 = conv(512, 256, kernel_size)
        # self.decode_conv3 = conv(256, 128, kernel_size)
        # self.decode_conv4 = conv(128, 64, kernel_size)
        # self.decode_conv5 = conv(64, 32, kernel_size)
        self.decode_conv1 = nn.Sequential(
            depthwise(1024, kernel_size),
            pointwise(1024, 512))
        self.decode_conv2 = nn.Sequential(
            depthwise(512, kernel_size),
            pointwise(512, 256))
        self.decode_conv3 = nn.Sequential(
            depthwise(256, kernel_size),
            pointwise(256, 128))
        self.decode_conv4 = nn.Sequential(
            depthwise(128, kernel_size),
            pointwise(128, 64))
        self.decode_conv5 = nn.Sequential(
            depthwise(64, kernel_size),
            pointwise(64, 32))
        self.decode_conv6 = pointwise(32, 1)
        weights_init(self.decode_conv1)
        weights_init(self.decode_conv2)
        weights_init(self.decode_conv3)
        weights_init(self.decode_conv4)
        weights_init(self.decode_conv5)
        weights_init(self.decode_conv6)

    def forward(self, x):
        # skip connections: dec4: enc1
        # dec 3: enc2 or enc3
        # dec 2: enc4 or enc5
        for i in range(14):
            layer = getattr(self, 'conv{}'.format(i))
            x = layer(x)
            # print("{}: {}".format(i, x.size()))
            if i==1:
                x1 = x
            elif i==3:
                x2 = x
            elif i==5:
                x3 = x
        for i in range(1,6):
            layer = getattr(self, 'decode_conv{}'.format(i))
            x = layer(x)
            x = F.interpolate(x, scale_factor=2, mode='nearest')
            if i==4:
                x = x + x1
            elif i==3:
                x = x + x2
            elif i==2:
                x = x + x3
            # print("{}: {}".format(i, x.size()))
        x = self.decode_conv6(x)
        return x

class MobileNetSkipConcat(nn.Module):
    def __init__(self, output_size, pretrained=True):

        super(MobileNetSkipConcat, self).__init__()
        self.output_size = output_size
        mobilenet = imagenet.mobilenet.MobileNet()
        if pretrained:
            pretrained_path = os.path.join('imagenet', 'results', 'imagenet.arch=mobilenet.lr=0.1.bs=256', 'model_best.pth.tar')
            checkpoint = torch.load(pretrained_path)
            state_dict = checkpoint['state_dict']

            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            mobilenet.load_state_dict(new_state_dict)
        else:
            mobilenet.apply(weights_init)

        for i in range(14):
            setattr( self, 'conv{}'.format(i), mobilenet.model[i])

        kernel_size = 5
        # self.decode_conv1 = conv(1024, 512, kernel_size)
        # self.decode_conv2 = conv(512, 256, kernel_size)
        # self.decode_conv3 = conv(256, 128, kernel_size)
        # self.decode_conv4 = conv(128, 64, kernel_size)
        # self.decode_conv5 = conv(64, 32, kernel_size)
        self.decode_conv1 = nn.Sequential(
            depthwise(1024, kernel_size),
            pointwise(1024, 512))
        self.decode_conv2 = nn.Sequential(
            depthwise(512, kernel_size),
            pointwise(512, 256))
        self.decode_conv3 = nn.Sequential(
            depthwise(512, kernel_size),
            pointwise(512, 128))
        self.decode_conv4 = nn.Sequential(
            depthwise(256, kernel_size),
            pointwise(256, 64))
        self.decode_conv5 = nn.Sequential(
            depthwise(128, kernel_size),
            pointwise(128, 32))
        self.decode_conv6 = pointwise(32, 1)
        weights_init(self.decode_conv1)
        weights_init(self.decode_conv2)
        weights_init(self.decode_conv3)
        weights_init(self.decode_conv4)
        weights_init(self.decode_conv5)
        weights_init(self.decode_conv6)

    def forward(self, x):
        # skip connections: dec4: enc1
        # dec 3: enc2 or enc3
        # dec 2: enc4 or enc5
        for i in range(14):
            layer = getattr(self, 'conv{}'.format(i))
            x = layer(x)
            # print("{}: {}".format(i, x.size()))
            if i==1:
                x1 = x
            elif i==3:
                x2 = x
            elif i==5:
                x3 = x
        for i in range(1,6):
            layer = getattr(self, 'decode_conv{}'.format(i))
            # print("{}a: {}".format(i, x.size()))
            x = layer(x)
            # print("{}b: {}".format(i, x.size()))
            x = F.interpolate(x, scale_factor=2, mode='nearest')
            if i==4:
                x = torch.cat((x, x1), 1)
            elif i==3:
                x = torch.cat((x, x2), 1)
            elif i==2:
                x = torch.cat((x, x3), 1)
            # print("{}c: {}".format(i, x.size()))
        x = self.decode_conv6(x)
        return x

class DenseNet(nn.Module):
    def __init__(self, layers, decoder, output_size, in_channels=3, pretrained=False):

        super(DenseNet, self).__init__()
        self.output_size = output_size
        
        densenet = torchvision.models.densenet121(pretrained=False)
        densenet_modules = list(densenet.features.children())
        
        if not pretrained:
            densenet.apply(weights_init)

        #densenet as encoder
        self.conv0 = nn.Conv2d(in_channels,64,kernel_size=7,stride=2,padding=3,bias=False)
        weights_init(self.conv0)
        self.norm0 = densenet_modules[1]
        self.relu0 = densenet_modules[2]
        self.maxpool0 = densenet_modules[3]
        self.denseblock1 = densenet_modules[4]
        self.transition1 = densenet_modules[5]
        self.denseblock2 = densenet_modules[6]
        self.transition2 = densenet_modules[7]
        self.denseblock3 = densenet_modules[8]
        self.transition3 = densenet_modules[9]
        self.denseblock4 = densenet_modules[10]
        self.norm1 = densenet_modules[11]
        #self.conv2 = nn.Conv2d(num_channels, 1024, 1)

        #decoder
        self.decoder = choose_decoder(decoder)       

        del densenet
        del densenet_modules

    def forward(self, x):
            x = self.conv0(x)
            #print('conv0 ', x.shape)
            x = self.norm0(x)
            #print('norm0 ', x.shape)
            x = self.relu0(x)
            #print('relu0 ', x.shape)
            x = self.maxpool0(x)
            #print('maxpool ', x.shape)
            x = self.denseblock1(x)
            #print('dense1 ', x.shape)
            x = self.transition1(x)
            #print('t1 ', x.shape)
            x = self.denseblock2(x)
            #print('d2 ', x.shape)
            x = self.transition2(x)
            #print('t2 ', x.shape)
            x = self.denseblock3(x)
            #print('d3 ', x.shape)
            x = self.transition3(x)
            #print('t3 ', x.shape)
            x = self.denseblock4(x)
            #print('d4 ', x.shape)
            x = self.norm1(x)
            #print('norm1', x.shape)

            x = self.decoder(x)
            #print('decoder ', x.shape)
            return x

class DenseNet_SkipAdd(nn.Module):
    def __init__(self, layers, output_size, in_channels=3, pretrained=False):

        super(DenseNet_SkipAdd, self).__init__()
        self.output_size = output_size
        
        densenet = torchvision.models.densenet121(pretrained=False)
        densenet_modules = list(densenet.features.children())
        
        if not pretrained:
            densenet.apply(weights_init)

        #densenet as encoder
        self.conv0 = nn.Conv2d(in_channels,64,kernel_size=7,stride=2,padding=3,bias=False)
        weights_init(self.conv0)
        self.norm0 = densenet_modules[1]
        self.relu0 = densenet_modules[2]
        self.maxpool0 = densenet_modules[3]
        self.denseblock1 = densenet_modules[4]
        self.transition1 = densenet_modules[5]
        self.denseblock2 = densenet_modules[6]
        self.transition2 = densenet_modules[7]
        self.denseblock3 = densenet_modules[8]
        self.transition3 = densenet_modules[9]
        self.denseblock4 = densenet_modules[10]
        self.norm1 = densenet_modules[11]
        #self.conv2 = nn.Conv2d(num_channels, 1024, 1)

        del densenet
        del densenet_modules

        #decoder
        #self.decoder = choose_decoder(decoder)
        kernel_size = 5
        if True :
            self.decode_conv1 = nn.Sequential(
                depthwise(1024, kernel_size),
                pointwise(1024, 512))
            self.decode_conv2 = nn.Sequential(
                depthwise(512, kernel_size),
                pointwise(512, 256))
            self.decode_conv3 = nn.Sequential(
                depthwise(256, kernel_size),
                pointwise(256, 128))
            self.decode_conv4 = nn.Sequential(
                depthwise(128, kernel_size),
                pointwise(128, 64))
            self.decode_conv5 = nn.Sequential(
                depthwise(64, kernel_size),
                pointwise(64, 32))
            self.decode_conv6 = pointwise(32, 1)

        weights_init(self.decode_conv1)
        weights_init(self.decode_conv2)
        weights_init(self.decode_conv3)
        weights_init(self.decode_conv4)
        weights_init(self.decode_conv5)
        weights_init(self.decode_conv6)

    def forward(self, x):
            #Densenet
            x = self.conv0(x)
            #print('conv0 ', x.shape)
            x = self.norm0(x)
            #print('norm0 ', x.shape)
            x = self.relu0(x)
            #print('relu0 ', x.shape)
            x4 = self.maxpool0(x)
            #print('maxpool ', x4.shape)
            x = self.denseblock1(x4)
            #print('dense1 ', x.shape)
            t1 = self.transition1(x)
            #print('t1 ', t1.shape)
            x = self.denseblock2(t1)
            #print('d2 ', x.shape)
            t2 = self.transition2(x)
            #print('t2 ', t2.shape)
            x = self.denseblock3(t2)
            #print('d3 ', x.shape)
            t3 = self.transition3(x)
            #print('t3 ', t3.shape)
            x = self.denseblock4(t3)
            #print('d4 ', x.shape)
            x = self.norm1(x)
            #print('norm1', x.shape)

            #decoder
            d1 = self.decode_conv1(x)
            #print('decoder1', d1.shape)
            #x = F.interpolate(d1 + t3, scale_factor=2, mode='nearest') no mem for this skip 
            x = F.interpolate(d1, scale_factor=2, mode='nearest')
            #print('interpolate1', x.shape)
            d2 = self.decode_conv2(x)
            #print('decoder2', d2.shape)
            x = F.interpolate(d2 + t2, scale_factor=2, mode='nearest')
            #print('interpolate2', x.shape)
            d3 = self.decode_conv3(x)
            #print('decoder3', d3.shape)
            x = F.interpolate(d3 + t1, scale_factor=2, mode='nearest')
            #print('interpolate3', x.shape)
            d4 = self.decode_conv4(x)
            #print('decoder4', d4.shape)
            x = F.interpolate(d4 + x4, scale_factor=2, mode='nearest')
            #print('interpolate4', x.shape)
            x = self.decode_conv5(x)
            #print('decoder5', x.shape)
            x = F.interpolate(x, scale_factor=2, mode='nearest')
            #print('interpolate5', x.shape)
            x = self.decode_conv6(x)
            #print('decoder6', x.shape)

            return x


class VGG(nn.Module):
    def __init__(self, layers, decoder, output_size, in_channels=3, pretrained=False):

        super(VGG, self).__init__()
        self.output_size = output_size
        
        vgg = torchvision.models.vgg11_bn(pretrained=False)
        vgg_modules = list(vgg.features.children())
        
        if not pretrained:
            vgg.apply(weights_init)

        self.conv0 = nn.Conv2d(in_channels,64,3,1,1)
        self.bn0   = vgg_modules[1] 
        self.relu0 = nn.ReLU(inplace=False)
        self.maxpool0 = vgg_modules[3]
        self.conv1 = vgg_modules[4]
        self.bn1   = vgg_modules[5]
        self.relu1 = nn.ReLU(inplace=False)
        self.maxpool1 = vgg_modules[7]
        self.conv2 = vgg_modules[8]
        self.bn2   = vgg_modules[9]
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = vgg_modules[11]
        self.bn3   = vgg_modules[12]
        self.relu3 = nn.ReLU(inplace=False)
        self.maxpool4 = vgg_modules[14]
        self.conv4 = vgg_modules[15]
        self.bn4   = vgg_modules[16]
        self.relu4 = nn.ReLU(inplace=False)
        self.conv5 = vgg_modules[18]
        self.bn5   = vgg_modules[19]
        self.relu5 = nn.ReLU(inplace=False)
        self.maxpool5 = vgg_modules[21]
        self.conv6 = vgg_modules[22]
        self.bn6   = vgg_modules[23]
        self.relu6 = nn.ReLU(inplace=False)
        self.conv7 = vgg_modules[25]
        self.bn7   = vgg_modules[26]
        self.relu7 = nn.ReLU(inplace=False)
        self.maxpool7 = vgg_modules[24]

        self.conv8 = nn.Conv2d(512,1024,1)

        #self.decoder = choose_decoder(decoder)
        kernel_size = 5
        if True :
            self.decode_conv1 = nn.Sequential(
                depthwise(1024, kernel_size),
                pointwise(1024, 512))
            self.decode_conv2 = nn.Sequential(
                depthwise(512, kernel_size),
                pointwise(512, 256))
            self.decode_conv3 = nn.Sequential(
                depthwise(256, kernel_size),
                pointwise(256, 128))
            self.decode_conv4 = nn.Sequential(
                depthwise(128, kernel_size),
                pointwise(128, 64))
            self.decode_conv5 = nn.Sequential(
                depthwise(64, kernel_size),
                pointwise(64, 32))
            self.decode_conv6 = pointwise(32, 1)

        weights_init(self.decode_conv1)
        weights_init(self.decode_conv2)
        weights_init(self.decode_conv3)
        weights_init(self.decode_conv4)
        weights_init(self.decode_conv5)
        weights_init(self.decode_conv6)


    def forward(self,x):
        #vgg11
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu0(x)
        x = self.maxpool0(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.maxpool4(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        x = self.maxpool5(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu6(x)
        x = self.conv7(x)
        x = self.bn7(x)
        x = self.relu7(x)
        x = self.maxpool7(x)
        #print(x.shape)

        #x = self.conv8(x)
        #print(x.shape)

        #x = self.decoder(x)
        #print(x.shape)

        #x = self.decode_conv1(x)
        #x = F.interpolate(x, scale_factor=2, mode='nearest')
        #print(x.shape)
        x = self.decode_conv2(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        #print(x.shape)
        x = self.decode_conv3(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        #print(x.shape)
        x = self.decode_conv4(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        #print(x.shape)
        x = self.decode_conv5(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        #print(x.shape)
        x = self.decode_conv6(x)
        #x = F.interpolate(x, scale_factor=2, mode='nearest')
        #print(x.shape)

        return x

class ShuffleNetV2(nn.Module):
    def __init__(self, decoder, output_size, in_channels=3, pretrained=False):

        super(ShuffleNetV2, self).__init__()
        self.output_size = output_size

        shufflenet = torchvision.models.shufflenet_v2_x1_0(pretrained=False)
        shufflenet_modules = list(shufflenet.children())

        if not pretrained:
            shufflenet.apply(weights_init)

        self.conv1 = nn.Conv2d(in_channels,24,kernel_size=3,stride=2,padding=1)
        self.bn1   = shufflenet_modules[0][1]  
        self.relu1 = shufflenet_modules[0][2]  
        self.maxpool1 = shufflenet_modules[1]
        self.stage2 = shufflenet_modules[2]
        self.stage3 = shufflenet_modules[3]
        self.stage4 = shufflenet_modules[4]
        self.conv5  = shufflenet_modules[5]

        self.decoder = choose_decoder(decoder)

        del shufflenet
        del shufflenet_modules

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        #print('after encoder ', x.shape)

        x = self.decoder(x)
        #print('after decoder ', x.shape)
        return x

class SqueezeNet(nn.Module):
    def __init__(self, decoder, output_size, in_channels=3, pretrained=False):

        super(SqueezeNet, self).__init__()
        self.output_size = output_size

        squeezenet = torchvision.models.squeezenet1_1(pretrained=False)
        squeezenet_modules = list(squeezenet.features.children())

        self.conv0 = nn.Conv2d(in_channels,64, kernel_size=3,stride=2)
        self.relu0 = squeezenet_modules[1]
        self.maxpool0 = squeezenet_modules[2]
        self.fire1 = squeezenet_modules[3]
        self.fire2 = squeezenet_modules[4]
        self.maxpool1 = squeezenet_modules[5]
        self.fire3 = squeezenet_modules[6]
        self.fire4 = squeezenet_modules[7]
        self.maxpool2 = squeezenet_modules[8]
        self.fire5 = squeezenet_modules[9]
        self.fire6 = squeezenet_modules[10]
        self.fire7 = squeezenet_modules[11]
        self.fire8 = squeezenet_modules[12]
       
        self.conv2 = nn.Conv2d(256,1024,1)

        self.decoder = choose_decoder(decoder)

    def forward(self,x):
        x = self.conv0(x)
        x = self.relu0(x)
        x = self.maxpool0(x)
        x = self.fire1(x)
        x = self.fire2(x)
        x = self.maxpool1(x)
        x = self.fire3(x)
        x = self.fire4(x)
        x = self.maxpool2(x)
        x = self.fire5(x)
        x = self.fire6(x)
        x = self.fire7(x)
        x = self.fire8(x)
        print("encoder ", x.shape)

        x = self.conv2(x)
        print("before decoder ", x.shape)

        x = self.decoder(x)
        return x


