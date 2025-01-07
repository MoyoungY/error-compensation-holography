import torch
import torchvision.models


class PerceptualLoss(torch.nn.modules.loss._Loss):

    def __init__(self, pixel_loss=1.0, l1_loss=False, style_loss=0.0, lambda_feat=1, include_vgg_layers=('1', '2', '3', '4', '5')):
        super(PerceptualLoss, self).__init__(True, True)

        # download pretrained vgg19 if necessary and instantiate it
        vgg19 = torchvision.models.vgg.vgg19(pretrained=True)
        self.vgg_layers = vgg19.features

        # the vgg feature layers we want to use for the perceptual loss
        self.layer_name_mapping = {
        }
        if '1' in include_vgg_layers:
            self.layer_name_mapping['3'] = "conv1_2"
        if '2' in include_vgg_layers:
            self.layer_name_mapping['8'] = "conv2_2"
        if '3' in include_vgg_layers:
            self.layer_name_mapping['13'] = "conv3_2"
        if '4' in include_vgg_layers:
            self.layer_name_mapping['22'] = "conv4_2"
        if '5' in include_vgg_layers:
            self.layer_name_mapping['31'] = "conv5_2"

        # weights for pixel loss and style loss (feature loss assumed 1.0)
        self.pixel_loss = pixel_loss
        self.l1_loss = l1_loss
        self.lambda_feat = lambda_feat
        self.style_loss = style_loss

    def forward(self, input, target):

        lossValue = torch.tensor(0.0).to(input.device)
        l2_loss_func = lambda ipt, tgt: torch.sum(torch.pow(ipt - tgt, 2))  # amplitude to intensity
        l1_loss_func = lambda ipt, tgt: torch.sum(torch.abs(ipt - tgt))  # amplitude to intensity

        # get size
        s = input.size()

        # number of tensors in this mini batch
        num_images = s[0]

        # L2 loss  (L1 originally)
        if self.l1_loss:
            scale = s[1] * s[2] * s[3]
            lossValue += l1_loss_func(input, target) * (2 * self.pixel_loss / scale)
            loss_func = l2_loss_func
        elif self.pixel_loss:
            scale = s[1] * s[2] * s[3]
            lossValue += l2_loss_func(input, target) * (2 * self.pixel_loss / scale)
            loss_func = l2_loss_func

        # stack input and output so we can feed-forward it through vgg19
        x = torch.cat((input, target), 0)

        for name, module in self.vgg_layers._modules.items():

            # run x through current module
            x = module(x)
            s = x.size()

            # scale factor
            scale = s[1] * s[2] * s[3]

            if name in self.layer_name_mapping:
                a, b = torch.split(x, num_images, 0)
                lossValue += self.lambda_feat * loss_func(a, b) / scale

                # Gram matrix for style loss
                if self.style_loss:
                    A = a.reshape(num_images, s[1], -1)
                    B = b.reshape(num_images, s[1], -1).detach()

                    G_A = A @ torch.transpose(A, 1, 2)
                    del A
                    G_B = B @ torch.transpose(B, 1, 2)
                    del B

                    lossValue += loss_func(G_A, G_B) * (self.style_loss / scale)

        return lossValue


import torch
import torchvision.models

class ModifiedPerceptualLoss(torch.nn.modules.loss._Loss):
    def __init__(self, pixel_loss=1.0, l1_loss=False, style_loss=0.0, lambda_feat=1, include_vgg_layers=('1', '2', '3', '4', '5')):
        super(ModifiedPerceptualLoss, self).__init__(True, True)

        # download pretrained vgg19 if necessary and instantiate it
        vgg19 = torchvision.models.vgg.vgg19(pretrained=True)
        self.vgg_layers = vgg19.features

        # the vgg feature layers we want to use for the perceptual loss
        self.layer_name_mapping = {
        }
        if '1' in include_vgg_layers:
            self.layer_name_mapping['3'] = "conv1_2"
        if '2' in include_vgg_layers:
            self.layer_name_mapping['8'] = "conv2_2"
        if '3' in include_vgg_layers:
            self.layer_name_mapping['13'] = "conv3_2"
        if '4' in include_vgg_layers:
            self.layer_name_mapping['22'] = "conv4_2"
        if '5' in include_vgg_layers:
            self.layer_name_mapping['31'] = "conv5_2"

        # weights for pixel loss and style loss (feature loss assumed 1.0)
        self.pixel_loss = pixel_loss
        self.l1_loss = l1_loss
        self.lambda_feat = lambda_feat
        self.style_loss = style_loss


    def forward(self, input, target, mask):
        # 扩展 target 和 mask 张量到三通道
        target = target.repeat(1, 3, 1, 1)

        total_loss = 0.0
        for i in range(input.shape[1]):  # 遍历所有 P 张图像
            # 选择第 i 张图像并扩展到三通道
            input_i = input[:, i, :, :].unsqueeze(1).repeat(1, 3, 1, 1)
            mask_i  = mask[:, i, :, :].unsqueeze(1).repeat(1, 3, 1, 1)
            # 计算第 i 张图像的损失
            loss_i = self._calculate_loss(input_i, target, mask_i)
            total_loss += loss_i

        # 计算平均损失
        average_loss = total_loss / input.shape[1]
        return average_loss

    def _calculate_loss(self, input, target, mask):
        # 初始化损失为0
        lossValue = torch.tensor(0.0).to(input.device)

        # 定义L1和L2损失函数
        l2_loss_func = lambda ipt, tgt, msk: torch.sum(torch.pow(ipt - tgt, 2) * msk)
        l1_loss_func = lambda ipt, tgt, msk: torch.sum(torch.abs(ipt - tgt) * msk)

        # 像素级损失
        if self.l1_loss:
            scale = mask.numel()
            lossValue += l1_loss_func(input, target, mask) * (self.pixel_loss / scale)
        elif self.pixel_loss:
            scale = mask.numel()
            lossValue += l2_loss_func(input, target, mask) * (self.pixel_loss / scale)

        # VGG特征损失
        x = torch.cat((input, target), 0)
        for name, module in self.vgg_layers._modules.items():
            x = module(x)

            if name in self.layer_name_mapping:
                a, b = torch.split(x, x.size(0) // 2, 0)
                # 为特征损失应用掩码
                feature_loss = l2_loss_func(a, b, mask)  # 通常使用L2损失计算特征损失
                scale = mask.numel()
                lossValue += self.lambda_feat * feature_loss / scale

        return lossValue
