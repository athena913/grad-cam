from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np

"""
Uses pretrained conv-nets to compute grad-cam.
Reference: https://github.com/kazuto1011/grad-cam-pytorch
"""

class GradCam(object):
      def __init__(self, model, target_layers):
        super(GradCam, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        #registered hooks (function handlers)
        self.hooks = []
        
        #maps layer to corresponding feature maps and gradients
        self.layer_fmaps = OrderedDict()
        self.layer_grads = OrderedDict() 
        self.target_layers = target_layers
        
        for name, module in self.model.named_modules():
            if target_layers is None or name in target_layers:
               self.hooks.append(module.register_forward_hook(self.forwardHook(name)))
               self.hooks.append(module.register_backward_hook(self.backwardHook(name)))
        
      def forwardHook(self,key):
          def forward_hook_(module, inp, out):
              #save the feature maps of the layer
              self.layer_fmaps[key] = out.detach()
          return forward_hook_    
      
      def backwardHook(self,key):
          def backward_hook_(module, grad_inp, grad_out):
              #save the gradients of the layer
              self.layer_grads[key] = grad_out[0].detach()
          return backward_hook_      
        
      def _encOneHot(self, ids):
          """ Generate a one-hot-encoding of the target class ids """
          
          one_hot = torch.zeros_like(self.logits).to(self.device)
          one_hot.scatter_(1, ids, 1.0)
          return one_hot
      
      def forward(self, x):
          
          """ Classification """
          self.img_shape = x.shape[2:]
          self.model.zero_grad()
          self.logits = self.model(x)
          probs = self.logits.softmax(dim=1)
          #print(self.logits.shape, probs.shape)
          probs = probs.sort(descending=True, dim=1)
          return self.logits, probs
      
      def backward(self, ids):
          """ Class-specific back prop """
          
          one_hot = self._encOneHot(ids)
          self.logits.backward(gradient=one_hot, retain_graph=True)
          
          return
      
      def _poolGrads(self, grads):
          
          pool = F.adaptive_avg_pool2d(grads, 1)
          #pool = nn.AdaptiveAvgPool2d(1)
          #return pool(grads) 
          return pool
      
      def genGCAM(self, layer):
         """ Generate grad-cam """
         
         if layer in self.layer_fmaps.keys():
            fmap = self.layer_fmaps[layer] 
         else: 
            raise ValueError("{} layer not found".format(layer))
         
         if layer in self.layer_grads.keys():
            grad = self.layer_grads[layer]  
         wts = self._poolGrads(grad)   
            
         gcam = torch.mul(fmap, wts).sum(dim=1, keepdim=True)
         print(gcam.shape)
         gcam = F.relu(gcam)   
         gcam = F.interpolate(gcam, self.img_shape, mode='bilinear', align_corners=False)
         
         B, C, H, W = gcam.shape
         gcam = gcam.view(B, -1)
         gcam -= gcam.min(dim=1, keepdim=True)[0]
         gcam /= gcam.max(dim=1, keepdim=True)[0]
         gcam = gcam.view(B, C, H, W)
         print(gcam.shape)
         return gcam
          
      def removeHooks(self):    
          """ Remove registered function handlers """
          
          for h in self.hooks:
              h.remove()


def getModel(dnn_model):
    """ Search for the specified pretrained model and return it """

    model_names = sorted(
      name
      for name in models.__dict__
      if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
     )
    #print (model_names)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.__dict__[dnn_model](pretrained=True)    
    model.to(device)
    model.eval()
    # use this  to get the layer names
    #mns = [m for m in model.named_modules()]
    #print(mns)
    return model, device



