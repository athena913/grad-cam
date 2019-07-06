This  is the readme file for the grad-cam project. 
The code is an implementation of Gradient-weighted class activation mapping. Convolutional feature maps are weighted by class-specific gradients to visualize features learned by a convolutional network.  The code uses convolutional image classifiers.

Dependencies: 

   Python2 or Python3

   Pytorch 1.0

   opencv

   matplotlib
   
Command line: python main_gcam.py

The code saves the grad-cam output in the specified folder. 
In main_gcam.py, set the input and output path in config according to your data and output folders. 



References:
1) Paper: Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, Dhruv Batra.https://arxiv.org/abs/1610.02391
2) Other implementations and tutorials:
   https://github.com/kazuto1011/grad-cam-pytorch
   
   https://medium.com/@stepanulyanin/grad-cam-for-resnet152-network-784a1d65f3
