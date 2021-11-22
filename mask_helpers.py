import torch
import torchvision
import numpy as np

def variance_map(x, threshold=0):
#   ''' creates the sigma-map for the batch x '''
    t_len = (4,) + x.shape
    t = torch.zeros(t_len).to(device='cuda:0')
    t[0,:,:-1] = x[:,1:]
    t[0,:,-1] = x[:,-1]
    t[1,:,1:] = x[:,:-1]
    t[1,:,0] = x[:,0]
    t[2,:,:,:-1] = x[:,:,1:]
    t[2,:,:,-1] = x[:,:,-1]
    t[3,:,:,1:] = x[:,:,:-1]
    t[3,:,:,0] = x[:,:,0]

    mean1 = (t[0] + x + t[1])/3
    sd1 = torch.sqrt(((t[0]-mean1)**2 + (x-mean1)**2 + (t[1]-mean1)**2)/3)

    mean2 = (t[2] + x + t[3])/3
    sd2 = torch.sqrt(((t[2]-mean2)**2 + (x-mean2)**2 + (t[3]-mean2)**2)/3)

    sd = torch.minimum(sd1, sd2)
    sd = torch.sqrt(sd)
    if(threshold>0):
        sd[sd<threshold] = 0
        sd[sd>threshold] = 1
        sd_sum = torch.sum(sd, dim=[1,2,3])
        for i in range(sd.shape[0]):
            if(sd_sum[i]==0):
                sd[i,:,:,:] += 1
                
    sd = (sd / torch.norm(sd, p=2, dim=[1,2,3]).view(sd.shape[0], 1, 1, 1)) * np.sqrt(sd.shape[1] * sd.shape[2] * sd.shape[3])

    return sd

def clip_var_map(x, norm, var_map):
    upper_lim = var_map * norm
    lower_lim = var_map * norm * (-1)
    x[x>upper_lim] = upper_lim[x>upper_lim]
    x[x<lower_lim] = lower_lim[x<lower_lim]    
#     print(torch.max(x))
#     print(torch.min(x))
    return x


def grad_mask(x, y, model, pixel_percent):
    # Calculate gradients of adv image
    delta = torch.zeros_like(x, requires_grad=True)
    loss = torch.nn.CrossEntropyLoss()(model(x + delta), y)
    loss.backward()
    
    # mask the grad by percentage
    flat_grad = delta.grad.detach().view(-1)
    mask_grad = torch.zeros_like(flat_grad)
    n = int(pixel_percent * len(flat_grad) / 100)
        
    max_idx = torch.argsort(torch.abs(flat_grad))[-n:]
    mask_grad[max_idx] = 1
    
    return mask_grad.view(x.shape)