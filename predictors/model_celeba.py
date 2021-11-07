# Adapted from https://github.com/Coderx7/A-Quick-and-Simple-Pytorch-Tutorial/blob/master/MultiTaskLearning.py
import torch
import torch.nn as nn
from torchvision import models

class ResNet50(nn.Module):
    def __init__(self, pretrained=True, frozen_feature_layers = False):
        super().__init__()
        
        resnet50 = models.resnet50(pretrained=pretrained)
        self.is_frozen = frozen_feature_layers
        self.features = nn.ModuleList(resnet50.children())[:-1]
        self.features = nn.Sequential(*self.features)

        if frozen_feature_layers:
            self.freeze_feature_layers()

        in_features = resnet50.fc.in_features
            
        self.fc_smile = nn.Linear(in_features, 2) 
        self.fc_age = nn.Linear(in_features, 2) 
        self.fc_gender = nn.Linear(in_features, 2) 
        self.fc_eyeglasses = nn.Linear(in_features, 2)


    def forward(self, input_imgs):
        output = self.features(input_imgs)
        output = output.view(input_imgs.size(0), -1)
        
        out_smile = self.fc_smile(output)
        out_age = self.fc_age(output)
        out_gender = self.fc_gender(output)
        out_glasses = self.fc_eyeglasses(output)
        
        return out_smile, out_age, out_gender, out_glasses
    
    def _set_freeze_(self, status):
        for n,p in self.features.named_parameters():
            p.requires_grad = status
        # for m in self.features.children():
        #     for p in m.parameters():
        #         p.requires_grad=status    


    def freeze_feature_layers(self):
        self._set_freeze_(False)

    def unfreeze_feature_layers(self):
        self._set_freeze_(True)