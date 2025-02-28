# from wilds import get_dataset
# from wilds.common.data_loaders import get_train_loader

import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt


from collections import defaultdict

import numpy as np
from prettytable import PrettyTable
from itertools import chain
from pathlib import Path
import collections
import json
import time
import csv


import torch
import pandas as pd
from tqdm import tqdm
from sconf import Config
from torch import nn
import collections


import os
import logging
import sys
import copy

from PIL import Image
from torch import optim
from transformers import CLIPProcessor, CLIPModel
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

import timm

import warnings

from domainbed.lib.logger import Logger

sys.path.append(os.path.abspath('../'))
from datasets import MultipleDomainDataset

import PlipTrain
# import Prompt.load_prompts as pt ## loading prompts


sys.path.append(os.path.abspath('./'))
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable tokenizers parallelism

warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.load.*weights_only=False.*")

# initialize the args dict
args = {} 


print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device Name:", torch.cuda.get_device_name(0))
    print("CUDA Version:", torch.version.cuda)
    print("Current Device:", torch.cuda.current_device())
else:
    print("CUDA not detected.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args['device'] = device


root_dir= "/home/user01/data"
args['data_dir'] = root_dir


transform = transforms.Compose(
    [
      transforms.Resize(224),
      transforms.ToTensor(),
    ]
)


root_data_dir = args['data_dir'] + '/camelyon17d'

print(root_dir)
train_dataset = MultipleDomainDataset(root_dir=root_data_dir, domains=['0', '1', '2'], transform=transform)
val_dataset = MultipleDomainDataset(root_dir=root_data_dir, domains=['3'], transform=transform)
test_dataset = MultipleDomainDataset(root_dir=root_data_dir, domains=['4'],  # Specify domains to include
    transform=transform
)

batch_size = 128
# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)


# Optional: Retrieve class and domain mappings
class_mapping = train_dataset.get_class_mapping()
domain_mapping = val_dataset.get_domain_mapping()
print("Class mapping:", class_mapping)
print("Domain mapping:", domain_mapping)



print(next(iter(train_loader))[0].shape)
sample_image = train_dataset[0][0]


dims = {
    # "RN50": 1024,
    "RN50": 512,
    "RN101": 512,
    "ViT-B/32": 512,
    "ViT-B/16": 512,
    "ViT-L/14": 768,
}


# Load the full dataset, and download it if necessary
# keys = ["config.yaml"]  #+ .configs  # Include default config file if needed

# keys = [open(key, encoding="utf8") for key in keys]
# args = Config(*keys, default=args)



args['gpu_id'] = 0
args['dataset_name'] = "camelyon17"
args['lambda'] = 0.5
args['prompt_sytle'] = 2


################# name of the version ####################
################# backbone of the student ###################
# args['name'] = "PLIP-VL2V-ADiP_Prompt_ViT_0"
args['backbone'] = "ViT-B/16"    # "ViT-L/14" "vit-B/16"
args['model'] = "vit-base"
#-----------------------------------------------------------#
# args['name'] = "PLIP-VL2V-ADiP_Prompt_RN_1"
# args['backbone'] = "RN50"    #\ resnet 50
# args['model'] = "resnet50"
#############################################################


args['pretrained'] = True  # student model must preserve its pretrained weights

# args['pretrained_path'] =  Path("./train_output") / args['dataset_name'] / args['name'] / 'models' 
# args["model"] = used for resnet

lr_list = np.array([5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3])

# args["optimizer"] = 'adamw'
args["optimizer"] = ("adam", "adam")
random_state = np.random.RandomState(0)
# args["lr"] = (5e-5, random_state.choice(lr_list))
args["lr"] = 5e-5
# args["weight_decay"] = (0.0, 10 ** random_state.uniform(-6, -2))
args["weight_decay"] = 10 ** random_state.uniform(-6, -2)
args['nsteps'] =  3000

args["work_dir"] = '.'
output_dir = args["work_dir"] / Path("train_output") 
output_dir.mkdir(exist_ok=True, parents=True)

args["out_root"] = args["work_dir"] / Path("train_output") / args["dataset_name"]
args["out_root"].mkdir(exist_ok=True, parents=True)

# args["out_dir"] = args["out_root"] / args["name"]
# args["out_dir"].mkdir(exist_ok=True, parents=True)

args["model_save"] = True
args["save_step"] = 200


# different settings for training: baseline vs ours
args["Prompt"] = False



# initialize data dict
data = {}
data['name'] = "camelyon17"
data['num_classes'] = 2
data['num_domains'] = 5
data['class_names'] = ['healthy', 'canserous']
print(sample_image.shape)
data['input_shape'] = sample_image.shape
# data['train_doms'] = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
# data['val_doms'] = [3, 2, 1, 0]
# data['train_doms'] = [0, 1, 2]
# data['val_doms'] = [3]
# data['test_doms'] = [4]



def save_records(records, output_root, filename="records.csv"):
    """
    Save the `records` to a CSV file.

    Args:
        records (list): The training/evaluation records.
        output_dir (str or Path): Directory to save the file.
        filename (str): Name of the output CSV file.
    """
    # Ensure output directory exists
    # output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Extract keys from the first record (assuming all records have the same keys)
    keys = records[0].keys()

    # Save to CSV file
    output_path = output_root / filename
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)

    print(f"Records saved to {output_path}")




# train_doms = [["0", "1", "2"], ["0", "1", "3"], ["0", "2", "3"], ["1", "2", "3"]]
# val_doms = ["3", "2", "1", "0"]

# domain 2 as test_domain
# train_doms = [["0", "1", "3"], ["0", "1", "4"], ["0", "3", "4"], ["1", "3", "4"]]
# val_doms = ["4", "3", "1", "0"]
# test_dom = ["2"]

# domain 0 as test_domain
train_doms = [["1", "2", "3"], ["1", "2", "4"], ["1", "3", "4"], ["2", "3", "4"]]
val_doms = ["4", "3", "2", "1"]
test_dom = ["0"]




transform = transforms.Compose(
    [
    transforms.Resize(224),
    transforms.ToTensor(),
    ]
)




batch_size = 128

def train_all_plip(args, data, train_doms, val_doms, transform, batch_size, checkpoint_freq, logger=None):
    """
    Shuffle domains and train the model on different train domains.

    Args:
        args (dict): Configuration arguments.
        data (dict): Dataset metadata.
        train_doms (list of lists): List of train domain combinations.
        val_doms (list): List of validation domains.
        transform (torchvision.transforms): Transformations for the dataset.
        checkpoint_freq (int): Frequency for saving checkpoints.
        logger (optional): Logger for logging training progress.
    """

    root_data_dir = args['data_dir'] + '/camelyon17d'

    root_dir= "/home/user01/data"
    args['data_dir'] = root_dir

    args['nsteps'] =  2500
    args['save_step'] = 100

    #test
    # args['nsteps'] =  300
    # args['save_step'] = 10
    # checkpoint_freq = 20

    # Iterate through shuffled train domains
    for _index, (train_domains, val_domain) in enumerate(zip(train_doms, val_doms)):

        data['train_doms'] = train_domains
        data['val_doms'] = val_domain
        data['test_doms'] = test_dom

        for i_stage in range(0, 2):  # changed this
            stage = i_stage + 1

            print(f"Stage {stage}: Training on domains {train_domains}, Validating on domain {val_domain}")

            root_data_dir = args['data_dir'] + '/camelyon17d'

            print(root_dir)
            train_dataset = MultipleDomainDataset(root_dir=root_data_dir, domains=train_domains, transform=transform)
            val_dataset = MultipleDomainDataset(root_dir=root_data_dir, domains=val_domain, transform=transform)
            # test_dataset = MultipleDomainDataset(root_dir=root_data_dir, domains=['4'], transform=transform)

            # Create DataLoader
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
            # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

            all_records = []
            # results = collections.defaultdict(list) #args['nsteps']
            res, records = PlipTrain.train(stage, train_loader=train_loader, val_loader=val_loader, args=args, data=data, checkpoint_freq=checkpoint_freq , logger=logger)
            
            train_dom_names = ''.join(map(str, data["train_doms"]))
            filename = "t_" + train_dom_names + "v_" + data["val_doms"] + f'_stage{stage}' + "records.csv"
            save_path = Path(args["out_root"]) / args["name"] / 'records'
            save_records(records, save_path, filename=filename)

            # # Optional: Retrieve class and domain mappings
            # class_mapping = train_dataset.get_class_mapping()
            # domain_mapping = val_dataset.get_domain_mapping()
            # print("Class mapping:", class_mapping)
            # print("Domain mapping:", domain_mapping)

            print(f"Stage {stage + 1} of training_dom:{i_stage}completed.\n")
            print("the result was: ", res)





def inference_all_plip(args, data, train_doms, val_doms, test_dom, transform, batch_size, logger=None, last = False):
    """
    Shuffle domains and train the model on different train domains.

    Args:
        args (dict): Configuration arguments.
        data (dict): Dataset metadata.
        train_doms (list of lists): List of train domain combinations.
        val_doms (list): List of validation domains.
        transform (torchvision.transforms): Transformations for the dataset.
        checkpoint_freq (int): Frequency for saving checkpoints.
        logger (optional): Logger for logging training progress.
    """



    root_data_dir = args['data_dir'] + '/camelyon17d'

    root_dir= "/home/user01/data"
    args['data_dir'] = root_dir

    results = []
    # Iterate through shuffled train domains
    
    for _index, (train_domains, val_domain) in enumerate(zip(train_doms, val_doms)):

        data['train_doms'] = train_domains
        data['val_doms'] = val_domain
        data['test_doms'] = test_dom

        stage = 3

        print(f"Stage {stage}: Training on domains {train_domains}, Validating on domain {val_domain}")

        root_data_dir = args['data_dir'] + '/camelyon17d'

        print(root_dir)
        test_dataset = MultipleDomainDataset(root_dir=root_data_dir, domains=test_dom, transform=transform)

        # Create DataLoader
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

        res, records = PlipTrain.inference(stage, test_loader=test_loader, args=args, data=data, logger=logger, last=last)
        results.append(res)

        train_dom_names = ''.join(map(str, data["train_doms"]))
        filename = "t_" + train_dom_names + "v_" + data["val_doms"] + f'_stage{stage}' + "test_records.csv"
        save_path = Path(args["out_root"]) / args["name"] / 'records'
        save_records(records, save_path, filename=filename)

        print(f"Inference of index:{_index} completed.\n")
        print("the result was: ", res)

    mean_results = defaultdict(list)
    for res in results:
        for key, value in res.items():
            mean_results[key].append(value)
    
    mean_results = {key: sum(values) / len(values) for key, values in mean_results.items()}


    print("Mean of results:", mean_results)
    if last:
        print("results for ========================= last models==============================")
    else:
        print("results for ========================= best models==============================")
    print('aggregated resluts:')
    for key in mean_results:
        logger.info(key + " is: " + str(mean_results[key]))


    train_dom_names = ''.join(map(str, data["train_doms"]))
    filename = "d_" + str(test_dom) + "_aggregated_test_records.csv"
    save_path = Path(args["out_root"]) / args["name"] / filename

    with open(save_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=mean_results.keys())
        
        # Write the header
        writer.writeheader()
        
        # Write the mean values
        writer.writerow(mean_results)

    print(f"Mean results saved to {save_path}")

    # filename = "aggregated_test_records.csv"
    # save_path = Path(args["out_root"]) / args["name"] / 'records'
    # save_records(mean_results, save_path, filename=filename)


    return mean_results


######################## using prelearned prompts ##############################
args["Prompt"] = False



args['name'] = "PLIP-VL2V-ADiP_ViT_0t_1"
args['backbone'] = "ViT-B/16"    # "ViT-L/14" "vit-B/16"
args['model'] = "vit-base"


# args['name'] = "PLIP-VL2V-ADiP_RN_0"
# args['backbone'] = "RN50"    #\ resnet 50
# args['model'] = "resnet50"

args["out_dir"] = args["out_root"] / args["name"]
args["out_dir"].mkdir(exist_ok=True, parents=True)
args['pretrained_path'] =  Path("./train_output") / args['dataset_name'] / args['name'] / 'models' 

logger = Logger.get(args["out_dir"] / "log.txt")

# inference_all_plip(args, data, train_doms, val_doms, test_dom, transform, batch_size, logger=logger)


train_all_plip(args, data, train_doms, val_doms, transform, batch_size, checkpoint_freq = 200, logger=logger)
