from torchvision.models.vgg import vgg16
import torch
import torch.nn as nn


class perception_loss(nn.Module):
    def __init__(self):
        super(perception_loss, self).__init__()
        features = vgg16(pretrained=True).features.cuda()

        self.to_relu_1_2 = nn.Sequential()
        self.to_relu_2_2 = nn.Sequential()
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False
        self.mse_loss = torch.nn.MSELoss()

    def forward(self, x, y):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        # out = (h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3)

        g = self.to_relu_1_2(y)
        g_relu_1_2 = g
        g = self.to_relu_2_2(g)
        g_relu_2_2 = g
        g = self.to_relu_3_3(g)
        g_relu_3_3 = g
        g = self.to_relu_4_3(g)
        g_relu_4_3 = g
        # out = (h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3)
        content_loss = self.mse_loss(
            h_relu_4_3, g_relu_4_3)
        return content_loss


class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        features = vgg16(pretrained=True).features.cuda()
        self.to_relu_1_2 = nn.Sequential()
        self.to_relu_2_2 = nn.Sequential()
        self.to_relu_3_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h

        out = (h_relu_1_2, h_relu_2_2, h_relu_3_3)
        return out

def gram(x):
    (bs, ch, h, w) = x.size()
    f = x.view(bs, ch, w * h)
    f_T = f.transpose(1, 2)
    G = f.bmm(f_T) / (ch * h * w)
    return G



class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.vgg = nn.DataParallel(Vgg16())
        self.vgg.eval()
        self.mse = nn.DataParallel(nn.MSELoss())
        self.mse_sum = nn.DataParallel(nn.MSELoss(reduction='sum'))

    def __call__(self, x, y_hat):
        style_features = self.vgg(x)
        y_hat_features = self.vgg(y_hat)

        style_gram = [gram(fmap) for fmap in style_features]
        y_hat_gram = [gram(fmap) for fmap in y_hat_features]
        L_style = 0
        for j in range(2):
            L_style += self.mse_sum(y_hat_gram[j], style_gram[j])
        return L_style


class content_style_loss():
    def __init__(self):
        self.content_loss = perception_loss()
        self.stype_loss = PerceptualLoss
    def forward(self, x, y_hat):
        content_loss = self.content_loss(x, y_hat)
        style_loss = self.content_loss(x, y_hat)
        return content_loss + style_loss
