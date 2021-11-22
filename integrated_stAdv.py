import torch, torchvision
from torchvision.datasets import ImageFolder
import numpy as np
import argparse
from PIL import Image
import kornia
from helpers.flow import Flow, flow_loss
from tqdm import tqdm
import random
from IQA_pytorch import SSIM, LPIPSvgg, DISTS
from xlwt import Workbook

from helpers.mask_helpers import variance_map, clip_var_map, grad_mask
from helpers.minimization_helpers import minimize_perturbation_batch, NormalizedNetwork, image_colorfulness
from helpers.cifar_nips_helpers import NIPS2017TargetedDataset, load_cifar_models, load_nips_data

load_cifar_models()
from PyTorch_CIFAR10.cifar10_models.resnet import resnet18, resnet34, resnet50
from PyTorch_CIFAR10.cifar10_models.densenet import densenet121, densenet161, densenet169
from PyTorch_CIFAR10.cifar10_models.inception import inception_v3

testloader = None
device = 'cuda:0'
    
def initilize_data(dataset_type="CIFAR10", batch_size=1, sample=None):

    global testloader
    if testloader == None:
        if(dataset_type=="CIFAR10"):
            transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                ]
            )
            testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                   download=True, transform=transform)

        elif(dataset_type=="imagenet"):
            load_nips_data()
            transform = torchvision.transforms.Compose([
#                 torchvision.transforms.Resize(256),
#                 torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
            ])
            
            testset = NIPS2017TargetedDataset("data/nips2017_targeted")


        if sample is not None:
            test_sample = list(range(sample))#random.sample(range(0, len(testset)), sample)
            testset = torch.utils.data.Subset(testset, test_sample)
        
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=1)
    else:
        pass
    return testset

def apply_flow (img, flow_layer, attack_YUV):
    if(attack_YUV == "RGB"):
        flowed_img = flow_layer(img)
        return flowed_img
    
    img_yuv = kornia.color.rgb_to_yuv(img)
    img_y, img_uv = img_yuv[:, :1, :, :], img_yuv[:, 1:, :, :]
    if attack_YUV == "Y only":
        flowed_img_y = flow_layer(img_y)
        flowed_img_yuv = torch.cat([flowed_img_y, img_uv], dim=-3)
    elif attack_YUV == "UV only":
        flowed_img_uv = flow_layer(img_uv)
        flowed_img_yuv = torch.cat([img_y, flowed_img_uv], dim=-3)
    else:
        flowed_img_y = flow_layer(img_y)
        flowed_img_uv = flow_layer(img_uv)
        flowed_img_yuv = torch.cat([flowed_img_y, flowed_img_uv], dim=-3)
    flowed_img = kornia.color.yuv_to_rgb(flowed_img_yuv)        
    return flowed_img


def UVAttack(images, labels, net, H, W, max_iteration, learning_rate, arrow_size, attack_YUV, target=None, num_random_init=1, init_method='normal', loss_type="ce", smoothness=0, rand_param=1, minimize=False, mask_type="None", pixel_percent=70):
    #already false prediction
    if(target == None and torch.argmax(torch.nn.functional.softmax(net(images), dim=1), axis=1) != labels):
        return images, 0, 0
    if(target != None and torch.argmax(torch.nn.functional.softmax(net(images), dim=1), axis=1) == target):
        return images, 0, 0

    global device

    success_iter = False
    loss_stack = []
    stack_range = 50
    successful_flow_img = None
    
    targets = (torch.ones_like(labels).to(device) * target) if (target!=None) else None
    
    for init_number in range(1, num_random_init+1):        
        if arrow_size > 0 :
            param_fn = lambda x: (torch.tanh(x) * arrow_size) 
        else:
            param_fn = None
        
        flow_layer = Flow(height=H, width=W, rand_param=rand_param, parameterization=param_fn, init_method=init_method).to(device)
        optimizer = torch.optim.Adam(flow_layer.parameters(), lr=learning_rate)
        
        for it in range(max_iteration):
            flowed_img_batch = apply_flow(images, flow_layer, attack_YUV)
            if mask_type == "var_clip":
                flowed_img_batch = images + clip_var_map(flowed_img_batch - images, 0.3, variance_map(images, threshold=0))
            elif mask_type == "var_weight":
                flowed_img_batch = images + (flowed_img_batch - images) * variance_map(images, threshold=0)
            elif mask_type == "grad":
                flowed_img_batch = images + (flowed_img_batch - images) * grad_mask(images, labels, net, pixel_percent)
            out = net(flowed_img_batch)
            
            if(loss_type=="cw" and target==None):
                loss = cw_loss(out, labels, False, confidence=0)
            elif(loss_type=="cw" and target!=None):
                loss = cw_loss(out, targets, True, confidence=0)
            elif(loss_type=="ce" and target==None):
                loss = -1 * torch.nn.functional.cross_entropy(out, labels)
            elif(loss_type=="ce" and target!=None):
                loss = torch.nn.functional.cross_entropy(out, targets)
#             print("Grad loss: {}, iter: {}".format(loss, it))

            if(smoothness>0 and it<max_iteration-5):
                loss += smoothness * flow_loss(flow_layer)
#                 print("Smooth loss: {}".format(smoothness * flow_loss(flow_layer)))

            #get last 10 (stack_range) loss to stack
            loss_stack.append(loss)
            while(len(loss_stack)>stack_range):
                loss_stack.pop(0)

                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #early stopping, if attack is successful
            if (smoothness == 0):
                if(target == None):
                    if(torch.argmax(torch.nn.functional.softmax(out, dim=1), axis=1) != labels):
                        success_iter = True
                        break
                else:
                    if(torch.argmax(torch.nn.functional.softmax(out, dim=1), axis=1) == target):
                        success_iter = True
                        break
            else:
                if(torch.argmax(torch.nn.functional.softmax(out, dim=1), axis=1) != labels):
                    successful_flow_img = flowed_img_batch
                    early_stopping_count = 0
                    if (loss_stack[0] - loss_stack[-1] < 0.1 and len(loss_stack) == stack_range):
                        success_iter = True
                        break

        if success_iter:
            break
    if(successful_flow_img != None):
        flowed_img_batch = successful_flow_img
    if minimize:
        flowed_img_batch = minimize_perturbation_batch(net, images, flowed_img_batch, labels, lr=1e-5, max_step_count=1000)
                        
    return flowed_img_batch, it+1, init_number

def cw_loss(logits, label, is_targeted=False, confidence=0):
    label = torch.nn.functional.one_hot(label, num_classes=logits.shape[1])
    assert logits.shape == label.shape
    real = torch.sum(label * logits, 1)
    other, _ = torch.max((1 - label) * logits - label, 1)
    f_fn = lambda real, other, targeted: torch.max(
        ((other - real) if targeted else (real - other)) + confidence,
        torch.tensor(0.0).to(real.device),
    )
    f = f_fn(real, other, is_targeted)
    loss = torch.sum(f) / len(label)
    
    return loss

    
def overall_testset_accuracy(wb, log_file, dataset_type="CIFAR10", iteration=100, learning_rate=0.1, arrow_size=10, target=None, attack_YUV="UV only", color_threshold=0, max_init=1, init_method='normal', network="resnet50", loss_type="ce", smoothness=0, rand_param=1, minimize=False, mask_type="None", pixel_percent=70):
    
    global device
    if(dataset_type=="CIFAR10"):
        H, W = 32, 32
        if(network=="resnet18"):
            net = resnet18(pretrained=True)
        elif(network=="resnet34"):
            net = resnet34(pretrained=True)
        elif(network=="resnet50"):
            net = resnet50(pretrained=True)
        elif(network=="densenet121"):
            net = densenet121(pretrained=True)
        elif(network=="densenet169"):
            net = densenet169(pretrained=True)

        net = NormalizedNetwork(net, mean=torch.tensor([0.4914, 0.4822, 0.4465]), std=torch.tensor([0.2471, 0.2435, 0.2616]), device=device)

    elif(dataset_type=="imagenet"):
#         H, W = 224, 224
        H, W = 299, 299

        if(network=="resnet18"):
            net = torchvision.models.resnet18(pretrained=True)
        elif(network=="resnet34"):
            net = torchvision.models.resnet34(pretrained=True)
        elif(network=="resnet50"):
            net = torchvision.models.resnet50(pretrained=True)
        elif(network=="densenet121"):
            net = torchvision.models.densenet121(pretrained=True)

        net = NormalizedNetwork(net, mean = torch.tensor([0.485, 0.456, 0.406]), std =  torch.tensor([0.229, 0.224, 0.225]), device=device)
    else:
        raise Exception('Invalid Dataset')
    
    net = net.to(device).eval()

    num_fooled = 0
    eliminated_testset_size = 0
    num_fooled_img = 0
    dataiter = iter(testloader)
    count=0
    iteration_count=0
    init_number = 0
    SCRSSIM = SSIM().to(device)
    SCRLPIPSvgg = LPIPSvgg().to(device)
    SCRDISTS = DISTS().to(device)
    
    rand_init_count=0
    ssim=0
    lpips=0
    dists=0
    l0=0
    l1=0
    l2=0
    linf=0

    if log_file is not None:
        sht1 = wb.add_sheet('Every Image Info')
        sht1.write(0, 0, 'Index')
        sht1.write(0, 1, "Is Fooled")
        sht1.write(0, 2, 'Total Iteration Count')
        sht1.write(0, 3, 'Random Init Count')
        sht1.write(0, 4, "SSIM")
        sht1.write(0, 5, 'LPIPS')
        sht1.write(0, 6, 'DISTS')
        sht1.write(0, 7, 'L0')
        sht1.write(0, 8, 'L1')
        sht1.write(0, 9, 'L2')
        sht1.write(0, 10, 'Linf')

        
    for data in tqdm(dataiter):
        if(dataset_type=="imagenet"):
            images, labels = data["image"], data["true_class"]
        else:
            images, labels = data[0], data[1]

        img_clrfl = image_colorfulness(images)
        is_fooled=False
        
        for i in range(images.shape[0]):
            if(img_clrfl[i]*255 > color_threshold):
                images, labels = images.to(device), labels.to(device)
                adv_img, it_count, init_count = UVAttack(images, labels, net, H, W, iteration, learning_rate, arrow_size, attack_YUV, target=target, num_random_init=max_init, init_method=init_method, loss_type=loss_type, smoothness=smoothness, rand_param=rand_param, minimize=minimize, mask_type=mask_type, pixel_percent=pixel_percent)
                results = torch.argmax(torch.nn.functional.softmax(net(adv_img), dim=1), axis=1)

                with torch.no_grad():
                    if (target == None):
                        if (results[i] != labels[i]):
                            is_fooled=True
                            if(init_count>1):
                                init_number += init_count
                                rand_init_count+=1

                    else:
                        if (results[i] == target):
                            is_fooled=True
                            if(init_count>1):
                                init_number += init_count
                                rand_init_count+=1

                eliminated_testset_size += 1
                iteration_count += it_count
                if (is_fooled):
                    ith_ssim  = SCRSSIM(images, adv_img, as_loss=False).cpu().item()
                    ith_lpips = SCRLPIPSvgg(images, adv_img, as_loss=False).cpu().item()
                    ith_dists = SCRDISTS(images, adv_img, as_loss=False).cpu().item()
                    ith_l0   = torch.dist(images, adv_img, p=0).cpu().detach().item()
                    ith_l1   = torch.dist(images, adv_img, p=1).cpu().detach().item()
                    ith_l2   = torch.dist(images, adv_img, p=2).cpu().detach().item()
                    ith_linf   = torch.dist(images, adv_img, p=float('inf')).cpu().detach().item()
                    ssim  += ith_ssim
                    lpips += ith_lpips
                    dists += ith_dists
                    l0    += ith_l0
                    l1    += ith_l1
                    l2    += ith_l2
                    linf  += ith_linf
                    num_fooled+=1
                else:
                    ith_ssim  = None
                    ith_lpips = None
                    ith_dists = None
                    ith_l0    = None
                    ith_l1    = None
                    ith_l2    = None
                    ith_linf  = None

                if log_file is not None:
                    sht1.write(count+1, 0, count)
                    sht1.write(count+1, 1, is_fooled)
                    sht1.write(count+1, 2, it_count)
                    sht1.write(count+1, 3, init_count)
                    sht1.write(count+1, 4, ith_ssim)
                    sht1.write(count+1, 5, ith_lpips)
                    sht1.write(count+1, 6, ith_dists)
                    sht1.write(count+1, 7, ith_l0)
                    sht1.write(count+1, 8, ith_l1)
                    sht1.write(count+1, 9, ith_l2)
                    sht1.write(count+1, 10, ith_linf)
                    wb.save(log_file)

        count+=1
            

    print("Fooling Rate: {}, Mean Iteration Count: {}, Eliminated Image Count: {}, Mean Random Init Count: {}".format(
                                                        num_fooled/eliminated_testset_size, 
                                                        iteration_count/eliminated_testset_size,
                                                        count - eliminated_testset_size, init_number/eliminated_testset_size))
    
    print("SSIM: {}, LPIPS: {}, DISTS: {}, L0: {}, L1: {}, L2: {}, Linf: {}".format(ssim/num_fooled, 
                                                  lpips/num_fooled, 
                                                  dists/num_fooled, l0/num_fooled, l1/num_fooled, l2/num_fooled, linf/num_fooled))
    net = net.cpu()
    del net
    return num_fooled/eliminated_testset_size, iteration_count/eliminated_testset_size, count - eliminated_testset_size, ssim/num_fooled, lpips/num_fooled, dists/num_fooled, l0/num_fooled, l1/num_fooled, l2/num_fooled, linf/num_fooled, str(init_number) + "/" + str(rand_init_count)

    
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='CIFAR10')
parser.add_argument('--attack_type', default='UV only')
parser.add_argument('--target', default=-1, type=int)
parser.add_argument('--iteration', default=100, type=int)
parser.add_argument('--arr_size', default=1, type=int)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--total_img_amount', default=-1, type=int)
parser.add_argument('--clr_thresh', default=15, type=int)
parser.add_argument('--max_init', default=1, type=int)
parser.add_argument('--log_file')
parser.add_argument('--init_method', default='normal')
parser.add_argument('--device', default='cuda:0')
parser.add_argument('--network', default='resnet50')
parser.add_argument('--loss_type', default='ce')
parser.add_argument('--smoothness', default=0.0, type=float)
parser.add_argument('--rand_param', default=1.0, type=float)
parser.add_argument('--minimize', default=False, type=bool)
parser.add_argument('--mask_type', default="None")
parser.add_argument('--pixel_percent', default=70, type=int)


args = parser.parse_args()

dataset_type=args.dataset
attack_type = args.attack_type
target=None if args.target < 0 else args.target
iteration=args.iteration
arr_size = int(args.arr_size)
lr=args.lr
total_img_amount = int(args.total_img_amount) if int(args.total_img_amount) > 0 else None
clr_thresh = args.clr_thresh
log_file = args.log_file
max_init_count = args.max_init
init_method = args.init_method
device = args.device
network = args.network
loss_type = args.loss_type
smoothness = args.smoothness
rand_param=args.rand_param
minimize=args.minimize
mask_type=args.mask_type
pixel_percent=args.pixel_percent

print(args)

wb = None
if log_file is not None:
    wb = Workbook(log_file)
    sht = wb.add_sheet('Sheet 1')

    param_idx = 20
    sht.write(param_idx-1, 0, 'Parameters')
    
    sht.write(param_idx, 0, 'DataSet Type')
    sht.write(param_idx, 1, 'Attack Type')
    sht.write(param_idx, 2, 'Target')
    sht.write(param_idx, 3, 'Arrow size')
    sht.write(param_idx, 5, 'Color Threshold')
    sht.write(param_idx, 4, 'Max Iteration')
    sht.write(param_idx, 6, 'Total Tested Images')
    sht.write(param_idx, 7, 'Max Random Init')
    sht.write(param_idx, 8, 'Init Method')
    sht.write(param_idx, 9, 'Network')
    sht.write(param_idx, 10, 'Loss Type')
    sht.write(param_idx, 11, 'Smoothness')
    sht.write(param_idx, 12, 'Random Init parameter')
    sht.write(param_idx, 13, 'Minimization')
    sht.write(param_idx, 14, 'Mask Type')
    sht.write(param_idx, 15, 'Pixel Percent')
    
    
    sht.write(param_idx+1, 0, dataset_type)
    sht.write(param_idx+1, 1, attack_type)
    sht.write(param_idx+1, 2, str(target))
    sht.write(param_idx+1, 3, arr_size)
    sht.write(param_idx+1, 5, clr_thresh)
    sht.write(param_idx+1, 4, iteration)
    sht.write(param_idx+1, 6, total_img_amount)
    sht.write(param_idx+1, 7, max_init_count)
    sht.write(param_idx+1, 8, init_method)
    sht.write(param_idx+1, 9, network)
    sht.write(param_idx+1, 10, loss_type)
    sht.write(param_idx+1, 11, smoothness)
    sht.write(param_idx+1, 12, rand_param)
    sht.write(param_idx+1, 13, minimize)
    sht.write(param_idx+1, 14, mask_type)
    sht.write(param_idx+1, 15, pixel_percent)

    sht.write(0, 0, 'Learning Rate')
    sht.write(0, 1, 'Fooling Rate')
    sht.write(0, 2, 'Mean Iteration Count')
    sht.write(0, 3, 'Eliminated Image Count')
    sht.write(0, 4, 'Mean Init Count')
    sht.write(0, 5, 'SSIM')
    sht.write(0, 6, 'LPIPS')
    sht.write(0, 7, 'DISTS')
    sht.write(0, 8, 'L0')
    sht.write(0, 9, 'L1')
    sht.write(0, 10, 'L2')
    sht.write(0, 11, 'Linf')

    indx=0

initilize_data(dataset_type=dataset_type, batch_size=1, sample=total_img_amount)

fooling_rt, mean_it_count, elm_img_count, ssim, lpips, dists, l0, l1, l2, linf, mean_init = overall_testset_accuracy(wb=wb, log_file=log_file, dataset_type=dataset_type, iteration=iteration, learning_rate=lr, arrow_size=arr_size, target=target, attack_YUV=attack_type, color_threshold=clr_thresh, max_init=max_init_count, init_method=init_method, network=network, loss_type=loss_type, smoothness=smoothness, rand_param=rand_param, minimize=minimize, mask_type=mask_type, pixel_percent=pixel_percent)



if log_file is not None:
    indx+=1
    sht.write(indx, 0, lr)
    sht.write(indx, 1, fooling_rt)
    sht.write(indx, 2, mean_it_count)
    sht.write(indx, 3, elm_img_count)
    sht.write(indx, 4, mean_init)
    sht.write(indx, 5, ssim)
    sht.write(indx, 6, lpips)
    sht.write(indx, 7, dists)
    sht.write(indx, 8, l0)
    sht.write(indx, 9, l1)
    sht.write(indx, 10, l2)
    sht.write(indx, 11, linf)

    wb.save(log_file)

# python3 integrated_stAdv.py --total_img_amount=100 --lr=0.005 --rand_param=0.1 --iteration=1000