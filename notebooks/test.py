#!/usr/bin/env python
# coding: utf-8

# In[1]:


from src.models.SwinUnet.networks.vision_transformer import SwinUnet

class Config:
    class DATA:
        IMG_SIZE = 128 
    
    class MODEL:
        DROP_RATE = 0.0  
        DROP_PATH_RATE = 0.1  
        LABEL_SMOOTHING = 0.1
        NAME = 'swin_tiny_patch4_window7_224'
        PRETRAIN_CKPT = 'src/models/SwinUnet/networks/swin_tiny_patch4_window7_224.pth'

        
        class SWIN:
            PATCH_SIZE = 4 
            IN_CHANS = 35  
            EMBED_DIM = 96  
            DEPTHS = [2, 2, 6, 2]  
            NUM_HEADS = [3, 6, 12, 24] 
            WINDOW_SIZE = 4  
            MLP_RATIO = 4.0  
            QKV_BIAS = True  
            QK_SCALE = None 
            APE = False
            PATCH_NORM = True

    class TRAIN:
        START_EPOCH = 0
        EPOCHS = 300
        WARMUP_EPOCHS = 20
        WEIGHT_DECAY = 0.05
        BASE_LR = 5e-4
        WARMUP_LR = 5e-7
        MIN_LR = 5e-6
        CLIP_GRAD = 5.0
        AUTO_RESUME = True
        ACCUMULATION_STEPS = 0
        USE_CHECKPOINT = False


        class LR_SCHEDULER:
            NAME = 'cosine'
            DECAY_EPOCHS = 30
            DECAY_RATE = 0.1

        class OPTIMIZER:
            NAME = 'adamw'
            EPS = 1e-8
            BETAS = (0.9, 0.999)
            MOMENTUM = 0.9
    
config = Config()

swin_unet = SwinUnet(config, num_classes=1)


# In[4]:


import torch 

for name, param in swin_unet.named_parameters():
    if param.dtype != torch.float32:
        print(f"Layer: {name}, Precision: {param.dtype}")



# In[26]:


import torch

from src.models.SwinUnet.networks.swin_transformer_unet_skip_expand_decoder_sys import PatchEmbed


patch_embedder = PatchEmbed()
inputs = torch.rand(1, 3, 224, 224)
embedding = patch_embedder(inputs)
print(f"when input size is {inputs.shape}, output size is: {embedding.shape}")

new_inputs = torch.rand(1, 3, 128, 128)
new_patch_embedder = PatchEmbed(img_size=128, patch_size=4)
new_embedding = new_patch_embedder(new_inputs)
print(f"when input size is {new_inputs.shape}, output size is: {new_embedding.shape}")




# In[17]:


inputs.shape


# In[13]:


import torch
import copy
pretrained_path = config.MODEL.PRETRAIN_CKPT

if pretrained_path is not None:
    print("pretrained_path:{}".format(pretrained_path))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pretrained_dict = torch.load(pretrained_path, map_location=device)
    pretrained_dict = pretrained_dict['model']
    print("---start load pretrained modle of swin encoder---")
    
    used_keys, unused_keys = [], []

    model_dict = swin_unet.state_dict()
    full_dict = copy.deepcopy(pretrained_dict)
    for k, v in pretrained_dict.items():
        if "layers." in k:
            current_layer_num = 3-int(k[7:8])
            current_k = "layers_up." + str(current_layer_num) + k[8:]
            full_dict.update({current_k:v})
    for k in list(full_dict.keys()):
        if k in model_dict:
            if full_dict[k].shape == model_dict[k].shape:
                used_keys.append(k)
            else:
                if "patch_embed.proj.weight" in k:
                        print(f"Updating patch embedding layer: {k}")
                        pretrained_weight = full_dict[k]
                        new_weight = torch.nn.init.xavier_uniform_(torch.empty_like(model_dict[k]))
                        num_pretrained_channels = pretrained_weight.shape[1]
                        new_weight[:, :num_pretrained_channels] = pretrained_weight[:, :num_pretrained_channels]
                        full_dict[k] = new_weight  # Use updated weights
                        used_keys.append(k)
                print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                del full_dict[k]
        else:
            unused_keys.append(k)
    msg = swin_unet.load_state_dict(full_dict, strict=False)
    print("Used pretrained keys:", used_keys)
    print("Unused pretrained keys:", unused_keys)
    print(msg)
else:
    print("none pretrain")


# In[2]:


model.load_from(config)


# In[2]:


import pandas as pd

def print_avg_test_AP(file_path, model_name):
    df = pd.read_csv(file_path)
    if 'data.features_to_keep' in df.columns:
        stats_ap_per_feature_set = df.groupby('data.features_to_keep')['test_AP'].agg(['mean', 'std'])
        print(model_name)
        print(stats_ap_per_feature_set.applymap(lambda x: f"{x:.3f}"))
    elif "optimizer.lr" in df.columns:
        stats_ap_per_feature_set = df.groupby("optimizer.lr")["test_AP"].agg(["mean", "std"])
        print(model_name)
        print(stats_ap_per_feature_set.applymap(lambda x: f"{x:.3f}"))

    else:
        overall_avg_ap = df['test_AP'].mean()
        overall_std_ap = df['test_AP'].std()
        print(model_name)
        print(f"Overall Average test_AP: {overall_avg_ap:.3f}")
        print(f"Overall Std test_AP: {overall_std_ap:.3f}")

       


# In[2]:


file_path = '/home/sl221120/WildfireSpreadTS/results/swin_mono_multi_all.csv'
model_name = "swin_monotemp_multi_all"
print_avg_test_AP(file_path, model_name)


# In[3]:


file_path = '/home/sl221120/WildfireSpreadTS/results/swin_mono_veg.csv'
model_name = "swin_monotemp_veg"
print_avg_test_AP(file_path, model_name)


# In[4]:


file_path = '/home/sl221120/WildfireSpreadTS/results/swin_multi_veg.csv'
model_name = "swin_multitemp_veg"
print_avg_test_AP(file_path, model_name)


# In[5]:


file_path = '/home/sl221120/WildfireSpreadTS/results/swin_multi_multi_all.csv'
model_name = "swin_multitemp_multi_all"
print_avg_test_AP(file_path, model_name)


# In[7]:


file_path = '/home/sl221120/WildfireSpreadTS/results/unet_mono_multi_all.csv'
model_name = "unet_monotemp_multi_all"
print_avg_test_AP(file_path, model_name)


# In[3]:


file_path = '/home/sl221120/WildfireSpreadTS/results/unet_multi_multi_all.csv'
model_name = "unet_multitemp_multi_all"
print_avg_test_AP(file_path, model_name)


# In[5]:


file_path = '/home/sl221120/WildfireSpreadTS/results/unet_multi_veg_lrs1.csv'
model_name = "unet_multitemp_veg_lrs_1"
print_avg_test_AP(file_path, model_name)


# In[4]:


file_path = '/home/sl221120/WildfireSpreadTS/results/unet_multi_veg_lrs2.csv'
model_name = "unet_multitemp_veg_lrs_2"
print_avg_test_AP(file_path, model_name)


# In[1]:


import os
import subprocess

# Define the base directory containing the runs
base_dir = '/home/sl221120/WildfireSpreadTS/lightning_logs/wildfire_progression'
data_dir = '/home/sl221120/scratch/WildfireSpreadTS_HDF5'

# Log data mapping run_id to data_fold_id
log_data = [
    {"run_id":"4adsj08p", "data_fold_id":0},
    {"run_id":"heg3b8sr", "data_fold_id":1},
    {"run_id":"g4kcwn56", "data_fold_id":2},
    {"run_id":"7t9l4h3a", "data_fold_id":3},
    {"run_id":"cgce7y3v", "data_fold_id":4},
    {"run_id":"h495vls8", "data_fold_id":5},
    {"run_id":"e90loozf", "data_fold_id":6},
    {"run_id":"81azfn75", "data_fold_id":7},
    {"run_id":"p8y7pgep", "data_fold_id":8},
    {"run_id":"a1cyprhl", "data_fold_id":9},
    {"run_id":"ju2yiuod", "data_fold_id":10},
    {"run_id":"w09hg1ub", "data_fold_id":11},
]

# Convert log data to a dictionary
run_id_to_fold_id = {entry["run_id"]: entry["data_fold_id"] for entry in log_data}

# Function to run the test command for a specific checkpoint
def run_test(checkpoint_path, data_fold_id):
    test_command = [
        'python',
        '/home/sl221120/WildfireSpreadTS/src/train.py',
        '--config=cfgs/Swin/swin.yaml',
        '--trainer=cfgs/trainer_single_gpu.yaml',
        '--data=cfgs/data_monotemporal_veg_features.yaml',
        '--seed_everything=0',
        #f'--data.data_fold_id={data_fold_id}',
        #f'--data.data_dir={data_dir}'
        f'--ckpt_path={checkpoint_path}'
    ]

    # Run the test command
    subprocess.run(test_command)

# Iterate through each run folder
# for run_id, data_fold_id in run_id_to_fold_id.items():
#     run_path = os.path.join(base_dir, run_id, 'checkpoints')
#     # Find the checkpoint file with the best validation loss
#     checkpoint_file = os.listdir(run_path)[0]
#     if checkpoint_file.endswith(".ckpt"):
#         run_test(checkpoint_file, data_fold_id)


# In[2]:


run_id = "k7s96l2o"
features_to_keep = [0, 1, 2, 3, 4, 38, 39]
run_path = os.path.join(base_dir, run_id, 'checkpoints')
checkpoint_file = os.listdir(run_path)[0]
checkpoint_path = (os.path.join(run_path, checkpoint_file))
checkpoint_path


# In[3]:


run_test(checkpoint_path=checkpoint_path, data_fold_id=0)


# In[3]:


log_data = [
    {"run_id":"4adsj08p", "data_fold_id":0},
    {"run_id":"heg3b8sr", "data_fold_id":1},
    {"run_id":"g4kcwn56", "data_fold_id":2},
    {"run_id":"7t9l4h3a", "data_fold_id":3},
    {"run_id":"cgce7y3v", "data_fold_id":4},
    {"run_id":"h495vls8", "data_fold_id":5},
    {"run_id":"e90loozf", "data_fold_id":6},
    {"run_id":"81azfn75", "data_fold_id":7},
    {"run_id":"p8y7pgep", "data_fold_id":8},
    {"run_id":"a1cyprhl", "data_fold_id":9},
    {"run_id":"ju2yiuod", "data_fold_id":10},
    {"run_id":"w09hg1ub", "data_fold_id":11},
]

# Convert log data to a dictionary
run_id_to_fold_id = {entry["run_id"]: entry["data_fold_id"] for entry in log_data}


run_id_to_fold_id.items()


# In[4]:


import pandas as pd 

def compute_ap(file_path):
    df = pd.read_csv(file_path)

    # Compute the average and standard deviation of test_AP
    test_ap_mean = df['test_AP'].mean()
    test_ap_std = df['test_AP'].std()


    print(test_ap_mean, test_ap_std)


# In[2]:


file_path = '/home/sl221120/WildfireSpreadTS/src/veg_mono.csv'
print("veg_mono")
compute_ap(file_path)


# In[3]:


file_path = '/home/sl221120/WildfireSpreadTS/src/multi_mono.csv'
print("multi_mono")
compute_ap(file_path)



# In[4]:


file_path = '/home/sl221120/WildfireSpreadTS/src/all_mono.csv'
print("all_mono")
compute_ap(file_path)


# In[6]:


file_path = '/home/sl221120/WildfireSpreadTS/src/veg_multi.csv'
print("veg_multi")
compute_ap(file_path)


# In[7]:


file_path = '/home/sl221120/WildfireSpreadTS/src/multi_multi.csv'
print("multi_multi")
compute_ap(file_path)


# In[6]:


file_path = '/home/sl221120/WildfireSpreadTS/src/all_multi.csv'
print("all_multi")
compute_ap(file_path)


# In[7]:


file_path = '/home/sl221120/WildfireSpreadTS/src/all_multi_new.csv'
print("all_multi")
compute_ap(file_path)


# In[ ]:


print("Veg Mono Unet FoldId0:")


