import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from IQA_pytorch import SSIM, LPIPSvgg, DISTS

class NormalizedNetwork(nn.Module):
    def __init__(
        self,
        base_model,
        mean=torch.tensor([0.4914, 0.4822, 0.4465]),
        std=torch.tensor([0.2471, 0.2435, 0.2616]),
        device="cuda:0",
    ):
        super(NormalizedNetwork, self).__init__()
        self.base_model = base_model.to(device)
        self.mean = mean.to(device)
        self.std = std.to(device)
        self.transform = transforms.Normalize(self.mean, self.std)

    def forward(self, x):
        x_norm = self.transform(x)
        return self.base_model(x_norm)

def color_conversion(x, mode="rgb to yuv", verbose=False):
    if mode not in ["rgb to yuv", "yuv to rgb"]:
        raise Exception("Invalid color mode")
    conversion_matrix = np.array(
        (
            [
                [0.299, 0.587, 0.114],
                [-0.14713, -0.28886, 0.436],
                [0.615, -0.51499, -0.10001],
            ]
        ),
        dtype=np.float32,
    )
    if mode == "yuv to rgb":
        conversion_matrix = np.linalg.inv(conversion_matrix)
    if isinstance(x, torch.Tensor):  # shape (1,3,W,H)
        conversion_matrix = torch.Tensor(conversion_matrix).to(x.device)
        #         print(x.dtype)
        result = (conversion_matrix @ x.float().view(3, -1)).view(x.shape)
    else:
        if x.size == 2:  # np.ndarray #shape (3,3)
            result = (conversion_matrix @ x.T).T.astype(x.dtype)
        else:  # W,H,3
            result = (
                (conversion_matrix @ x.T.reshape(3, -1))
                .T.reshape(x.shape)
                .astype(x.dtype)
            )
    return result

def image_colorfulness(image, return_color_averages=True):
    if not isinstance(image, torch.Tensor):
        image = torch.from_numpy(image)

    # split the image into its respective RGB components
    R = image[:, 0, :, :].reshape((image.shape[0], -1))
    G = image[:, 1, :, :].reshape((image.shape[0], -1))
    B = image[:, 2, :, :].reshape((image.shape[0], -1))
    # compute rg = R - G
    rg = torch.absolute(R - G)
    # compute yb = 0.5 * (R + G) - B
    yb = torch.absolute(0.5 * (R + G) - B)
    # compute the mean and standard deviation of both `rg` and `yb`
    (rbMean, rbStd) = (torch.mean(rg, axis=1), torch.std(rg, axis=1))

    (ybMean, ybStd) = (torch.mean(yb, axis=1), torch.std(yb, axis=1))
    # combine the mean and standard deviations
    stdRoot = torch.sqrt((rbStd ** 2) + (ybStd ** 2))
    meanRoot = torch.sqrt((rbMean ** 2) + (ybMean ** 2))
    # derive the "colorfulness" metric and return it
    if return_color_averages:
        return stdRoot + (0.3 * meanRoot), torch.mean(R), torch.mean(G), torch.mean(B)
    else:
        return stdRoot + (0.3 * meanRoot)


def michelson_contrast(image, return_mean_y=False):
    yuv_image = color_conversion(image, mode="rgb to yuv")
    Y = yuv_image[:, 0, :, :].reshape((image.shape[0], -1))

    lum_max, _ = torch.max(Y, dim=1)
    lum_min, _ = torch.min(Y, dim=1)
    #     print(lum_max)
    #     print(lum_min)

    if not return_mean_y:
        return (lum_max - lum_min) / (lum_max + lum_min)
    else:
        return (lum_max - lum_min) / (lum_max + lum_min), torch.mean(Y, dim=1)


def yuv_dist_fn(x_adv, x, y_weight, uv_weights, norm=2):
    
    yuv_x_adv = color_conversion(x_adv, mode="rgb to yuv", verbose=False)
    yuv_x = color_conversion(x, mode="rgb to yuv", verbose=False)
  
    perturbation = (yuv_x_adv - yuv_x).reshape(len(x), 3, -1)
    
    a = (perturbation ** 2).sum(dim=2).sqrt()
    
    y_loss = (a[:, 0])*y_weight
    
    uv_loss = (a[:, 1] + a[:, 2])*uv_weights
    
    total_loss = y_loss + uv_loss
    
    return total_loss.mean()


def tanh(x, bias=0, scale=1):
    x = x * scale
    return bias + (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))


def sigmoid_01(x, scale=1, beta=1):
    x = x * scale
    return 1 / (1 + (x / (1 - x)) ** -beta)


def sigmoid(x, bias=0, scale=1):
    return (1 / (1 + torch.exp(-scale * x))) + bias


def minimize_perturbation_batch(
    model, image, adv_image, label, lr=1e-3, max_step_count=30, yuv=False, dist_type="l2"
):
    #     print('------------------')
    if dist_type == "lpips":
        SCRLPIPSvgg = LPIPSvgg().cuda(0)
    model_out = model(adv_image)
    pred = model_out.argmax(dim=1)
    correct_count = (pred == label).sum().detach()

    opt_adv_ex = torch.clone(adv_image).detach()
    opt_adv_ex.requires_grad = True
    best = torch.clone(adv_image).detach()
    colorfulness = image_colorfulness(image, False)
    contrast = michelson_contrast(image)
    optimizer = torch.optim.Adam([opt_adv_ex], lr)
#     optimizer = torch.optim.Adam([opt_adv_ex])
    for i in range(max_step_count):
        model_out = model(opt_adv_ex)
        pred = model_out.argmax(dim=1)

        if (pred == label).sum() > correct_count:
            if dist_type == "lpips":
                SCRLPIPSvgg = SCRLPIPSvgg.cpu()
                del SCRLPIPSvgg
            return best
        best = opt_adv_ex.clone()

        if not yuv:
            if dist_type == "l2":
                dist = torch.dist(opt_adv_ex, image, p=2)                
            elif dist_type == "lpips":
                dist = SCRLPIPSvgg(opt_adv_ex, image, as_loss=True)

        else:
            dist = yuv_dist_fn(
                opt_adv_ex, image, y_weight=contrast, uv_weights=colorfulness, norm=2
            )

        dist.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    model_out = model(opt_adv_ex)
    pred = model_out.argmax(dim=1)

    if (pred == label).sum() > correct_count:
        if dist_type == "lpips":
            SCRLPIPSvgg = SCRLPIPSvgg.cpu()
            del SCRLPIPSvgg
        return best
    
    if dist_type == "lpips":
        SCRLPIPSvgg = SCRLPIPSvgg.cpu()
        del SCRLPIPSvgg
    return opt_adv_ex

