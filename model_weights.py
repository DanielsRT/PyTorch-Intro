import torch
import torchvision.models as models

##Saving and Loading Model Weights

#Pytorch models store the learned parameters in an internal state dictionary, called state_dict. 
# These can be persisted via the torch.save method:

model = models.vgg16(weights='IMAGENET1K_V1')
torch.save(model.state_dict(), 'vgg_model_weights.pth')

#To load model weights, you need to create an instance of the same model first, and then load the 
# parameters using the load_state_dict method.

#In the code below, we set weights_only=True to limit the functions executed during unpickling to 
# only those necessary for loading weights. Using weights_only=True is considered best practice 
# when loading weights.

model = models.vgg16() # we do not specify "weights", i.e. create untrained model
model.load_state_dict(torch.load('vgg_model_weights.pth', weights_only=True))
model.eval()

##Saving and Loading Models with Shapes

#When loading model weights, we need to instantiate the model first, because the class defines the 
# scruture of a network. We might want to save the structure of this class together with the model, 
# in which case we can pass model (and not model.state_dict()) to the torch.save method.

torch.save(model, 'vgg_model.pth')

#Saving state_dict is considered best practice. However bewlow we use weights_only=False because this 
# involves loading the model, which is a legacy use case for torch.save.

model = torch.load('vgg_model.pth', weights_only=False),