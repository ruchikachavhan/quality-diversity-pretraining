import torch
import math
import copy
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

class BranchedResNet(nn.Module):
    def __init__(self, N, arch, num_classes, stop_grad = True, clip = False):
        super(BranchedResNet, self).__init__()
        if arch == 'resnet50':
            #  Load ImageNet1k_V2 weights
            self.base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            self.num_feat = 2048
        # add one to self.N to account for the baseline model
        self.N = N + 1
        self.num_classes = num_classes


        # Branching out only one Resnet50 layer, need to instantiate another Resnet50 model due to some deep copy issues
        self.base_model.branches_layer4 = nn.ModuleList([resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).layer4 for _ in range(self.N)])
        self.base_model.branches_fc = nn.ModuleList([resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).fc for _ in range(self.N)])

        del self.base_model.layer4, self.base_model.fc

        if stop_grad:
            for name, param in self.base_model.named_parameters():
                if 'layer4' not in name and 'fc' not in name:
                    param.requires_grad = False
                if param.requires_grad:
                    print("Learning param: ", name, param.requires_grad)
                else:
                    print("Freezing param: ", name, param.requires_grad)
        
        # Freezing gradients of baseline model in ensemble, which is the last model of the ensemble
        for name, param in self.base_model.branches_layer4[-1].named_parameters():
            param.requires_grad = False
            print("Freezing layer4 of baseline branch: ", name, param.requires_grad)
        
        for name, param in self.base_model.branches_fc[-1].named_parameters():
            param.requires_grad = False
            print("Freezing fc of baseline branch: ", name, param.requires_grad)

    def forward(self, x, reshape = True):
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)
        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        feats = [self.base_model.avgpool(self.base_model.branches_layer4[i](x)).view(x.shape[0], -1) for i in range(self.N)]
        outputs = [self.base_model.branches_fc[i](feats[i]) for i in range(self.N)]

        if reshape:
            outputs = torch.cat(outputs).reshape(self.N, -1, self.num_classes)
            feats = torch.cat(feats).reshape(self.N, -1, self.num_feat)

        return outputs, feats