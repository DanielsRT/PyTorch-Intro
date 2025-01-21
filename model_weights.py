import torch
import torchvision.models as models

##Saving and Loading Model Weights

#Pytorch models store the learned parameters in an internal state dictionary, called state_dict. These can be persisted via the torch.save method:

model = models.vgg16(weights='IMAGENET1K_V1')
torch.save(model.state_dict(), 'model_weights.pth')