#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
from torch.utils.data import DataLoader
from src.dataloader.FireSpreadDataset import FireSpreadDataset
import copy

data_dir = "/home/sl221120/scratch/WildfireSpreadTS_HDF5"
included_fire_years = [2017, 2018, 2019]
n_leading_observations = 1
crop_side_length = 128
load_from_hdf5 = True
is_train = True
remove_duplicate_features = True
stats_years = (2018, 2019)
n_leading_observations_test_adjustment = 5
features_to_keep = [0, 1, 2, 3, 4, 38, 39]
return_doy = False
desired_side_length = 224

dataset = FireSpreadDataset(
    data_dir=data_dir,
    included_fire_years=included_fire_years,
    n_leading_observations=n_leading_observations,
    crop_side_length=crop_side_length,
    load_from_hdf5=load_from_hdf5,
    is_train=is_train,
    remove_duplicate_features=remove_duplicate_features,
    stats_years=stats_years,
    n_leading_observations_test_adjustment=n_leading_observations_test_adjustment,
    features_to_keep=features_to_keep,
    return_doy=return_doy,
    desired_side_length=desired_side_length
)

# Create DataLoader
data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

for batch in data_loader:
    inputs, label = batch
    break



# In[3]:


class Config:
    class DATA:
        IMG_SIZE = 224 
    
    class MODEL:
        TYPE = "swin"
        DROP_RATE = 0
        DROP_PATH_RATE = 0.2
        NAME = 'swin_tiny_patch4_window7_224'
        PRETRAIN_CKPT = 'src/models/SwinUnet/networks/swin_tiny_patch4_window7_224.pth'

        
        class SWIN:
            PATCH_SIZE = 4 
            IN_CHANS = 7  
            EMBED_DIM = 96  
            DEPTHS = [2, 2, 2, 2]
            DECODER_DEPTHS = [2, 2, 2, 1]  
            NUM_HEADS = [3, 6, 12, 24] 
            WINDOW_SIZE = 7
            MLP_RATIO = 4.0  
            QKV_BIAS = True  
            QK_SCALE = None 
            APE = False
            PATCH_NORM = True   
        
    class TRAIN:
        USE_CHECKPOINT = True

config = Config()


# In[4]:


from src.models.SwinUnet.networks.vision_transformer import SwinUnet

model = SwinUnet(config, num_classes=1)


# In[5]:


from torchviz import make_dot
from IPython.display import Image

x = torch.randn(1, 7, 224, 224)
y = model(x)
#make_dot(y, params=dict(model.named_parameters()))


# In[6]:


pretrained_path = config.MODEL.PRETRAIN_CKPT
if pretrained_path is not None:
    print("pretrained_path:{}".format(pretrained_path))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pretrained_dict = torch.load(pretrained_path, map_location=device)
    pretrained_dict = pretrained_dict['model']

pretrained_dict.keys()


# In[7]:


for k in model.state_dict().keys():
    if not k.startswith("swin_unet"):
        print(k)


# In[96]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pretrained_dict = torch.load(pretrained_path, map_location=device)
pretrained_dict['model'].keys()


# In[97]:


for k in pretrained_dict['model'].keys():
    print(k)


# In[16]:


from collections import OrderedDict

def load_from(model, config):
    pretrained_path = config.MODEL.PRETRAIN_CKPT
    if pretrained_path is not None:
        print("pretrained_path:{}".format(pretrained_path))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pretrained_dict = torch.load(pretrained_path, map_location=device)
        if "model" not in pretrained_dict:
            print("---start load pretrained modle by splitting---")
            pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}
            for k in list(pretrained_dict.keys()):
                if "output" in k:
                    print("delete key:{}".format(k))
                    del pretrained_dict[k]
            msg = model.load_state_dict(pretrained_dict,strict=False)
            # print(msg)
            return
        pretrained_dict = pretrained_dict['model']
        print("---start load pretrained modle of swin encoder---")

        model_dict = model.state_dict()
        
        new_pretrained_dict = OrderedDict()
        for k in pretrained_dict.keys():
            new_key = 'swin_unet.' + k  # rename key
            new_pretrained_dict[new_key] = pretrained_dict[k]

        full_dict = copy.deepcopy(new_pretrained_dict)
        for k, v in pretrained_dict.items():
            if "layers." in k:
                current_layer_num = 3-int(k[7:8])
                current_k = "swin_unet.layers_up." + str(current_layer_num) + k[8:]
                full_dict.update({current_k:v})
        i = 0
        matched = []
        for k in list(full_dict.keys()):
            if k in model_dict:
                i += 1
                matched.append(k)
                if full_dict[k].shape != model_dict[k].shape:
                    print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                    del full_dict[k]
        msg = model.load_state_dict(full_dict, strict=False)
        #print(msg)
        print(f"Number of keys loaded: {len(model_dict) - len(msg.missing_keys)}")
        print(f"Number of keys missing: {len(msg.missing_keys)}")
        print(f"Missing keys: {msg.missing_keys}")
        print(f"Number of unexpected keys: {len(msg.unexpected_keys)}")
        print(f"Unexpected keys: {msg.unexpected_keys}")
    else:
        print("none pretrain")


# In[17]:


load_from(model, config)


# In[34]:


# Get the keys from the model's state dict and the pretrained dict
model_keys = set(model.state_dict().keys())
pretrained_keys = set(pretrained_dict.keys())

# Find missing keys in the model that are in the pretrained dict
missing_in_model = pretrained_keys - model_keys
print(len(missing_in_model))

# Find missing keys in the pretrained dict that are in the model
missing_in_pretrained = model_keys - pretrained_keys
print(len(missing_in_pretrained))

same_keys = pretrained_keys.intersection(model_keys)
print(len(same_keys))


# In[ ]:





# In[28]:


same_keys


# In[4]:


model.swin_unet.patch_embed.proj.weight.shape


# In[5]:


pretrained_path = 'src/models/SwinUnet/networks/swin_tiny_patch4_window7_224.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pretrained_dict = torch.load(pretrained_path, map_location=device)
pretrained_dict = pretrained_dict['model']
pretrained_dict.keys()


# In[6]:


import copy
model_dict = model.state_dict()
full_dict = copy.deepcopy(pretrained_dict)
# 190 keys
for k, v in pretrained_dict.items():
    if "layers." in k:
        current_layer_num = 3-int(k[7:8])
        current_k = "layers_up." + str(current_layer_num) + k[8:]
        full_dict.update({current_k:v})
# 372 keys
for k in list(full_dict.keys()):
    if k in model_dict:
        if full_dict[k].shape != model_dict[k].shape:
            print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
            del full_dict[k]

msg = model.load_state_dict(full_dict, strict=False)
print(msg)


# In[7]:


full_dict.keys()


# In[ ]:





# In[37]:


print(model_dict["swin_unet.patch_embed.proj.weight"].shape)
print(full_dict["patch_embed.proj.weight"].shape)


# In[24]:


n_features = FireSpreadDataset.get_n_features(n_leading_observations, features_to_keep, remove_duplicate_features)
old_conv = model.swin_unet.patch_embed.proj

new_conv = nn.Conv2d(n_features, old_conv.out_channels, 
                     kernel_size=old_conv.kernel_size, 
                     stride=old_conv.stride, 
                     padding=old_conv.padding, 
                     bias=old_conv.bias is not None)

with torch.no_grad():
    new_conv.weight[:,:3,:,:] = old_conv.weight
    if n_features > 3:
        nn.init.kaiming_normal_(new_conv.weight[:, 3:, :, :], mode='fan_out', nonlinearity='relu')    
    if old_conv.bias is not None:
        print("Old conv bias is not None. Replacing new bias with learned one.")
        new_conv.bias = old_conv.bias
model.swin_unet.patch_embed.proj = new_conv


# In[25]:


# Test with a dummy input to ensure it works
dummy_image = torch.randn(1, n_features, 224, 224) 
output = model(dummy_image)
print(output.shape)  #


# In[2]:


import torch
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import SwinForImageClassification, SwinConfig, SwinModel
import torch.nn.functional as F

# Import your dataset and datamodule
from src.dataloader.FireSpreadDataModule import FireSpreadDataModule
from src.dataloader.FireSpreadDataset import FireSpreadDataset

# Define your own model class by extending the pretrained Swin model
class SwinFineTuner(pl.LightningModule):
    def __init__(self):
        super(SwinFineTuner, self).__init__()
        config = SwinConfig.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
        self.model = SwinModel(config)
        
        # Add a final convolutional layer to generate a binary mask
        self.classifier = nn.Conv2d(config.hidden_size, 1, kernel_size=1)
        
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x):
        outputs = self.model(x).last_hidden_state
        outputs = outputs.permute(0, 3, 1, 2)  # Change shape to [batch, hidden_size, height, width]
        logits = self.classifier(outputs)
        return logits

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        self.log('val_loss', loss, prog_bar=True)
        return loss

# Parameters
batch_size = 16

# Initialize the data module
data_module = FireSpreadDataModule(batch_size=batch_size)

# Initialize the model
model = SwinFineTuner()

# Set up a model checkpoint callback
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath="./checkpoints",
    filename="swin-transformer-{epoch:02d}-{val_loss:.2f}",
    save_top_k=1,
    mode="min",
)

# Initialize the trainer
trainer = Trainer(
    max_epochs=1,
    gpus=1,  # Use GPU if available
    callbacks=[checkpoint_callback]
)



# In[ ]:


# Train the model
trainer.fit(model, datamodule=data_module)

