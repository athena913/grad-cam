import sys
import os
import glob

from PIL import Image
import matplotlib.cm as cm

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset 
import cv2
import numpy as np
#following is needed only for models trained on Imagenet data
from keras.applications.imagenet_utils import decode_predictions

import grad_cam as gcam

"""
Gradient-CAM 
"""

def visGCam(hm, ipath, opath):
  """
  Visualize the grad-cam
  """

  im = cv2.imread(ipath)
  hm = cv2.resize(hm, (im.shape[1], im.shape[0]))
  hm = np.uint8(255* hm)
  hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
  overlay = hm*0.4 + im
  cv2.imwrite(opath, overlay)
  return

def saveGCam(gc, raw_image, opath, paper_cmap=False):
    """
    Save the grad-cam overlaid on the image 
    """
    gc = gc.cpu().numpy()
    cmap = cm.jet_r(gc)[..., :3] * 255.0
    if paper_cmap:
        alpha = gc[..., None]
        gc = alpha * cmap + (1 - alpha) * raw_image
    else:
        gc = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2
    cv2.imwrite(opath, np.uint8(gc))

    return

def getTopkPreds(logits, k=5):
    
   #get the most likely class prediction
   #pred = logits.argmax(dim=1)

   #get the top-5 class predictions
   preds = logits.argsort(dim=1, descending=True)[:,:k]
   preds = preds.squeeze().tolist()
   #get the human readable labels from keras function
   klogs = logits.data.numpy()
   labels = decode_predictions(klogs)
   print(preds, labels)
   return preds, labels

def compCam(dl):
    
    sample = next(iter(dl))
    cam.computeCAM(sample)

    return    
 
def computeGCam(dnn_model, target_layers,  dl, out_path, top_k=5):
    """ 
    Get the specified pretrained model,
    Obtain the predicted logits for the input images,
    Compute the gradient CAM wrt the top-5 class predictions,
    Save and visualize the grad-CAMs overlaid on the original images.
    """
    model, device = gcam.getModel(dnn_model)
    n = 0

    #load the next batch
    sample = next(iter(dl))
    imgs = sample["x"].to(device)
    img_list = sample["fn"]
        
    #load the raw images to overlay  the gradient maps    
    raw_images = []
    for j, img_path in enumerate(img_list):
        raw_image =  cv2.imread(img_path)
        raw_images.append(cv2.resize(raw_image, (224, 224)))
              
   
    gc = gcam.GradCam(model, target_layers)
   
    #forward prop to get the conv feature maps and classify the images
    logits, sort_probs = gc.forward(imgs)
    #convert preds to class names
    labels = decode_predictions(logits.cpu().data.numpy())
    print(labels)
    #extract the sorted class posterior probabilities and ids
    prob, cids = sort_probs    
    
    #output class activation maps for top k predictions for each file
    for k in range(top_k):
        #back-prop to output class activation maps for top k predictions 
        gc.backward(cids[:, [k]])

        for t in target_layers: 
            #compute the gradient maps from the specified layers 
            grad_map = gc.genGCAM(t)
            print(grad_map.shape)
            for j in range(len(img_list)):
              fn, _ = os.path.splitext(img_list[j].split("/")[-1])  
              op = os.path.join(out_path, "gcam_%s_%s_%s_%s.jpg"%(fn, t, str(k), labels[j][k][1]))
              #overlay the gradient map on the corresponding raw image and save it.
              saveGCam(grad_map[j, 0], raw_images[j], op)
    return

        
    

class ImgDataset(Dataset):

    """ iterates over the file list and 
        returns a batch of (image, filename) pairs.
    """

    def __init__(self, im_path, transform = None):

       # get the list of img files in the specified folder 
       self.files = [] 
       for f in sorted(glob.glob(im_path + "/*.jpg")): 
           self.files.append(f)
           
       print("loaded meta-data:{}".format(len(self.files)))
       self.transform = transform

    def __len__(self):
        
       """ return length of the dataset """
    
       return len(self.files)

    def __getitem__(self, n):
        
       im_path = self.files[n]
       x = Image.open(im_path)

       print(im_path)
       #apply data transform if required
       if self.transform is not None:
          x = self.transform(x)
          
       #im_path is used for displaying test data results
       sample = {"x": x, "fn": im_path}
       return sample
   

def getImageLoader(im_path, bs):   
   """ Get a data loader that loads images from the specified path """

   #data transforms
   transform = transforms.Compose([transforms.Resize((224, 224)),
               transforms.ToTensor(),
               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


   images = ImgDataset(im_path, transform)
   print("Num images={}".format(len(images)))
   data_loader = DataLoader(dataset=images, shuffle=False, batch_size=bs, num_workers=0)
   return data_loader

if __name__ == "__main__":

   """ Modify the config parameters as per your needs """
   config = {
              "img_path"   : "./data/",
              "batch_size" : 2, #batch size
              "model_name" : "resnet18",
              "out_dir"   : "vis_gcam"
            }
   #specify the layers for feature extraction for each model
   model_layers = {"vgg19": ["features.35"], 
                   "resnet18": ["layer4"]}
   model_name = config["model_name"]

   data_loader = getImageLoader(config["img_path"], config["batch_size"])

   out_dir = os.path.join(config["out_dir"], model_name)
   if not os.path.exists(out_dir):
      os.makedirs(out_dir)
      
   computeGCam(model_name, model_layers[model_name], data_loader, out_dir)
   
