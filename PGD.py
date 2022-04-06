import torch
import torch.nn as nn

def pgd_attack(model, images, labels, criterion, eps=0.3, alpha=2/255, iters=40):
    ori_images = images.data
    for i in range(iters) :    
        images.requires_grad = True
        outputs = model(images)
        outputs = outputs[0] + outputs[1] + outputs[2]

        model.zero_grad()
        cost = criterion(outputs, labels)
        cost.backward()

        adv_images = images + alpha * images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()
            
    return images