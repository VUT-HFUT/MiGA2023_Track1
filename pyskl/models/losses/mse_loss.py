import torch.nn.functional as F
import torch
import torch.nn as nn
from ..builder import LOSSES

def mse_center_loss(output, target, labels):
    #(out['embed'], embeddings, target)
    t = labels.clone().detach()
    t[t >= 0.5] = 1  # threshold to get binary labels
    t[t < 0.5] = 0


    positive_centers = []
    for i in range(output.size(0)):
        p = target[i]
        if p.size(0) == 0:
            positive_center = torch.zeros(300).cuda()
        else:
            # positive_center = torch.mean(p, dim=0)
            positive_center = p
        positive_centers.append(positive_center)

    positive_centers = torch.stack(positive_centers,dim=0)
    #在dim=0上堆积
    loss = F.mse_loss(output, positive_centers)
  
    return loss


@LOSSES.register_module()
class MseLoss(nn.Module):
    def forward(self, output, target, labels):
        loss = mse_center_loss(output, target, labels)
        return loss