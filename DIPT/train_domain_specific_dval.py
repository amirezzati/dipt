import os
import sys
import json
import argparse
from collections import OrderedDict
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import torchvision.transforms as transforms
import torchvision.models as models

from tqdm import tqdm
from PIL import Image

from transformers import CLIPProcessor, CLIPModel

from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# Add project-specific paths

sys.path.append(os.path.abspath('/home/user/dg'))
sys.path.append(os.path.abspath('/home/user/dg/Prompt/KG/KgCoOp/OurUtils'))
sys.path.append(os.path.abspath('/home/user/dg/utils'))
sys.path.append(os.path.abspath('/home/user/dg/Prompts'))

from avg_template_prompts import get_text_feature_kather
from prompts_utils import train_with_domain_prompt_diff_val,zero_shot_classification
from datasets import MultipleDomainDataset


def main():
    parser = argparse.ArgumentParser(description="Train a model using domain-specific prompts.")

    # Add arguments
    parser.add_argument("--classnames", nargs="+", default=["normal", "tumor"], help="List of class names.")
    parser.add_argument("--domain_num", nargs="+", default=["0", "1"], help="Domain numbers to use for training.")
    parser.add_argument("--val_domain_num", nargs="+", default=["1"], help="val Domain numbers to use for training.")
    parser.add_argument("--gpu_index", type=int, default=4, help="GPU index to use.")
    parser.add_argument("--num_context_tokens", type=int, default=4, help="Number of learnable context tokens.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate for the optimizer.")
    parser.add_argument("--score_weight", type=float, default=0.5, help="Weight for the score in the loss function.")
    # New flag to choose the learnable agg_vector version
    parser.add_argument("--use_learnable_agg", action="store_true", help="Use learnable aggregate vector.")
    parser.add_argument("--eval_interval", type=int, default=50, help="Number of batches between validation evaluations.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--prefix", type=str, default="default_prefix", help="A prefix string for the experiment.")

    # Parse arguments
    args = parser.parse_args()

    # Setup device
    gpu_index = args.gpu_index
    if torch.cuda.is_available() and gpu_index < torch.cuda.device_count():
        device = torch.device(f"cuda:{gpu_index}")
        print(f"Using device: {device} - {torch.cuda.get_device_name(gpu_index)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Define root directory and domains
    root_dir = '/home/user/data/kather100k/train'
    domains = args.domain_num  # Correct, makes ['0', '1']

    # Define transformations (modify as needed)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Create datasets and dataloaders
    print(domains)
    train_set = MultipleDomainDataset(root_dir, domains, transform=transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    
    
    val_set = MultipleDomainDataset(root_dir, args.val_domain_num, transform=transform)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True)
    

    # Define PLIP model and its processor
    plip = CLIPModel.from_pretrained("vinid/plip")
    plip_processor = CLIPProcessor.from_pretrained("vinid/plip")
    plip.to(device)

    # Create the aggregate text feature (agg_vector)
    agg_text_feature_template = get_text_feature_kather(plip_processor, plip, device)
    agg_text_feature_template /= agg_text_feature_template.norm(dim=-1, keepdim=True)
    agg_text_feature_template = agg_text_feature_template.to(device)
    agg_vector = agg_text_feature_template  

    domain_num = args.domain_num
    
    eval_interval = int(len(train_loader)*0.09)

    # Call the training function with the extra flag
    train_with_domain_prompt_diff_val(
        plip=plip,
        plip_processor=plip_processor,
        train_loader=train_loader,
        classnames=args.classnames,
        domain_num=domain_num,
        device=device,
        val_loader=val_loader,
        num_context_tokens=args.num_context_tokens,
        agg_vector=agg_vector,
        num_epochs=args.num_epochs,
        lr=args.lr,
        score_weight=args.score_weight,
        use_learnable_agg=args.use_learnable_agg , # new flag passed here
        eval_interval=eval_interval,  # Pass eval_interval here
        prefix=args.prefix
    )
    

if __name__ == "__main__":
    main()
