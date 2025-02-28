import sys
import os

import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import OrderedDict


from datetime import datetime

import torchvision.transforms as transforms


sys.path.append(os.path.abspath('/home/user/dg/Prompts'))
from prompts_utils import PromptLearner, TextEncoder

from transformers import CLIPProcessor, CLIPModel




def get_learned_prompts(classnames, plip, plip_processor, num_context_tokens=4, doms=None, device="cuda"):
    
    # Define domain paths in a dictionary
    DOMAIN_PATHS = {
        0: "/home/user/dg/Prompts/output_kg/0/2025-02-18_18-28-09/best_prompt_learner.pth",
        1: "/home/user/dg/Prompts/output_kg/1/2025-02-18_21-11-52/best_prompt_learner.pth",
        2: "/home/user/dg/Prompts/output_kg/2/2025-02-18_18-28-29/best_prompt_learner.pth",
        3: "/home/user/dg/Prompts/output_kg/3/2025-02-18_18-28-35/best_prompt_learner.pth",
        4: "/home/user/dg/Prompts/output_kg/4/2025-02-18_19-57-44/best_prompt_learner.pth"
    }

    # Validate requested domains
    doms = [int(d) for d in doms] if doms else list(DOMAIN_PATHS.keys())
    invalid_doms = [d for d in doms if d not in DOMAIN_PATHS]
    if invalid_doms:
        raise ValueError(f"Invalid domains requested: {invalid_doms}")

    # Load all requested domain prompt learners
    prompt_learners = []
    for domain in doms:
        # Load checkpoint
        checkpoint = torch.load(DOMAIN_PATHS[domain])
        
        # Create and configure prompt learner
        pl = PromptLearner(classnames, plip, plip_processor, num_context_tokens)
        pl.load_state_dict(checkpoint)
        pl.to(device)
        prompt_learners.append(pl)

    # Process all prompts in batch
    text_encoder = TextEncoder(plip).to(device)
    
    # Collect all prompts and tokens
    all_prompts = []
    all_tokens = []
    for pl in prompt_learners:
        with torch.no_grad():
            prompts = pl()
            all_prompts.append(prompts)
            all_tokens.append(pl.tokenized_learnable_prompts)

    # Encode all text features in one batch
    text_features = torch.stack([
        text_encoder(p, t) 
        for p, t in zip(all_prompts, all_tokens)
    ])

    # Calculate mean of requested domains
    mean_prompts = text_features.mean(dim=0)
    mean_prompts = mean_prompts / mean_prompts.norm(dim=-1, keepdim=True)
    mean_prompts = mean_prompts.detach().clone()

    # Print debug information
    # print(f"Loaded domains: {doms}")
    # print(f"Learned prompts shape: {text_features.shape}")
    # print(f"Mean prompts shape: {mean_prompts.shape}")

    return mean_prompts



def get_learned_prompts_patient(classnames, plip, plip_processor, num_context_tokens=4, device="cuda"):
    
    # Define domain paths in a dictionary
    DOMAIN_PATHS = {
        0: "/home/user/dg/Prompts/Patient_based/output_kg/patient12/2025-02-22_20-26-55/best_prompt_learner.pth",
        1: "/home/user/dg/Prompts/Patient_based/output_kg/patient15/2025-02-22_21-03-26/best_prompt_learner.pth",
        2: "/home/user/dg/Prompts/Patient_based/output_kg/patient21/2025-02-22_19-37-34/best_prompt_learner.pth",
        3: "/home/user/dg/Prompts/Patient_based/output_kg/patient23/2025-02-22_18-37-06/best_prompt_learner.pth",
        4: "/home/user/dg/Prompts/Patient_based/output_kg/patient26/2025-02-22_20-16-26/best_prompt_learner.pth",
        5: "/home/user/dg/Prompts/Patient_based/output_kg/patient28/2025-02-22_21-40-42/best_prompt_learner.pth",
        6: "/home/user/dg/Prompts/Patient_based/output_kg/patient45/2025-02-22_19-18-59/best_prompt_learner.pth",
        7: "/home/user/dg/Prompts/Patient_based/output_kg/patient58/2025-02-22_20-44-29/best_prompt_learner.pth",
        8: "/home/user/dg/Prompts/Patient_based/output_kg/patient59/2025-02-22_21-30-28/best_prompt_learner.pth",
        9: "/home/user/dg/Prompts/Patient_based/output_kg/patient61/2025-02-22_19-08-07/best_prompt_learner.pth",
        10: "/home/user/dg/Prompts/Patient_based/output_kg/patient70/2025-02-22_19-56-27/best_prompt_learner.pth",
        11: "/home/user/dg/Prompts/Patient_based/output_kg/patient81/2025-02-22_21-21-49/best_prompt_learner.pth",
        12: "/home/user/dg/Prompts/Patient_based/output_kg/patient82/2025-02-22_21-13-10/best_prompt_learner.pth",
        13: "/home/user/dg/Prompts/Patient_based/output_kg/patient88/2025-02-22_20-06-33/best_prompt_learner.pth",
        14: "/home/user/dg/Prompts/Patient_based/output_kg/patient96/2025-02-22_18-57-14/best_prompt_learner.pth",
        15: "/home/user/dg/Prompts/Patient_based/output_kg/patient110/2025-02-22_18-48-03/best_prompt_learner.pth",
        16: "/home/user/dg/Prompts/Patient_based/output_kg/patient114/2025-02-22_20-54-29/best_prompt_learner.pth",
        17: "/home/user/dg/Prompts/Patient_based/output_kg/patient128/2025-02-22_19-47-55/best_prompt_learner.pth",
        18: "/home/user/dg/Prompts/Patient_based/output_kg/patient145/2025-02-22_20-35-41/best_prompt_learner.pth",
        19: "/home/user/dg/Prompts/Patient_based/output_kg/patient160/2025-02-22_19-29-03/best_prompt_learner.pth",
    }


    # Load all requested domain prompt learners
    prompt_learners = []
    for domain in range(20):
        # Load checkpoint
        checkpoint = torch.load(DOMAIN_PATHS[domain])
        
        # Create and configure prompt learner
        pl = PromptLearner(classnames, plip, plip_processor, num_context_tokens)
        pl.load_state_dict(checkpoint)
        pl.to(device)
        prompt_learners.append(pl)

    # Process all prompts in batch
    text_encoder = TextEncoder(plip).to(device)
    
    # Collect all prompts and tokens
    all_prompts = []
    all_tokens = []
    for pl in prompt_learners:
        with torch.no_grad():
            prompts = pl()
            all_prompts.append(prompts)
            all_tokens.append(pl.tokenized_learnable_prompts)

    # Encode all text features in one batch
    text_features = torch.stack([
        text_encoder(p, t) 
        for p, t in zip(all_prompts, all_tokens)
    ])

    # Calculate mean of requested domains
    mean_prompts = text_features.mean(dim=0)
    mean_prompts = mean_prompts / mean_prompts.norm(dim=-1, keepdim=True)
    mean_prompts = mean_prompts.detach().clone()

    # Print debug information
    # print(f"Loaded domains: {doms}")
    # print(f"Learned prompts shape: {text_features.shape}")
    # print(f"Mean prompts shape: {mean_prompts.shape}")

    return mean_prompts


def get_learned_prompts_val1(classnames, plip, plip_processor, num_context_tokens=4, doms=None, device="cuda"):
    
    # Define domain paths in a dictionary
    DOMAIN_PATHS = {
        0: "/home/user/dg/Prompts/output_kg/diff_val/0/2025-02-19_23-00-54/best_prompt_learner.pth",
        2: "/home/user/dg/Prompts/output_kg/diff_val/2/2025-02-20_01-57-52/best_prompt_learner.pth",
        3: "/home/user/dg/Prompts/output_kg/diff_val/3/2025-02-19_23-00-40/best_prompt_learner.pth",
        4: "/home/user/dg/Prompts/output_kg/diff_val/4/2025-02-19_23-00-38/best_prompt_learner.pth"
    }

    # Validate requested domains
    doms = [int(d) for d in doms] if doms else list(DOMAIN_PATHS.keys())
    invalid_doms = [d for d in doms if d not in DOMAIN_PATHS]
    if invalid_doms:
        raise ValueError(f"Invalid domains requested: {invalid_doms}")

    # Load all requested domain prompt learners
    prompt_learners = []
    for domain in doms:
        # Load checkpoint
        checkpoint = torch.load(DOMAIN_PATHS[domain])
        
        # Create and configure prompt learner
        pl = PromptLearner(classnames, plip, plip_processor, num_context_tokens)
        pl.load_state_dict(checkpoint)
        pl.to(device)
        prompt_learners.append(pl)

    # Process all prompts in batch
    text_encoder = TextEncoder(plip).to(device)
    
    # Collect all prompts and tokens
    all_prompts = []
    all_tokens = []
    for pl in prompt_learners:
        with torch.no_grad():
            prompts = pl()
            all_prompts.append(prompts)
            all_tokens.append(pl.tokenized_learnable_prompts)

    # Encode all text features in one batch
    text_features = torch.stack([
        text_encoder(p, t) 
        for p, t in zip(all_prompts, all_tokens)
    ])

    # Calculate mean of requested domains
    mean_prompts = text_features.mean(dim=0)
    mean_prompts = mean_prompts / mean_prompts.norm(dim=-1, keepdim=True)
    mean_prompts = mean_prompts.detach().clone()

    # Print debug information
    # print(f"Loaded domains: {doms}")
    # print(f"Learned prompts shape: {text_features.shape}")
    # print(f"Mean prompts shape: {mean_prompts.shape}")

    return mean_prompts






def get_learned_prompts_kather(classnames, plip, plip_processor, num_context_tokens=4, doms=None, device="cuda"):
    
    # Define domain paths in a dictionary
    DOMAIN_PATHS = {
        0: "/home/user/dg/Prompts/output_kg/kather/0/2025-02-25_21-32-08/best_prompt_learner.pth",
        1: "/home/user/dg/Prompts/output_kg/kather/1/2025-02-25_21-55-12/best_prompt_learner.pth",
        2: "/home/user/dg/Prompts/output_kg/kather/2/2025-02-25_22-12-21/best_prompt_learner.pth",
        3: "/home/user/dg/Prompts/output_kg/kather/3/2025-02-25_22-26-23/best_prompt_learner.pth",
        4: "/home/user/dg/Prompts/output_kg/kather/4/2025-02-25_23-04-49/best_prompt_learner.pth"
    }

    # Validate requested domains
    doms = [int(d) for d in doms] if doms else list(DOMAIN_PATHS.keys())
    invalid_doms = [d for d in doms if d not in DOMAIN_PATHS]
    if invalid_doms:
        raise ValueError(f"Invalid domains requested: {invalid_doms}")

    # Load all requested domain prompt learners
    prompt_learners = []
    for domain in doms:
        # Load checkpoint
        checkpoint = torch.load(DOMAIN_PATHS[domain])
        
        # Create and configure prompt learner
        pl = PromptLearner(classnames, plip, plip_processor, num_context_tokens)
        pl.load_state_dict(checkpoint)
        pl.to(device)
        prompt_learners.append(pl)

    # Process all prompts in batch
    text_encoder = TextEncoder(plip).to(device)
    
    # Collect all prompts and tokens
    all_prompts = []
    all_tokens = []
    for pl in prompt_learners:
        with torch.no_grad():
            prompts = pl()
            all_prompts.append(prompts)
            all_tokens.append(pl.tokenized_learnable_prompts)

    # Encode all text features in one batch
    text_features = torch.stack([
        text_encoder(p, t) 
        for p, t in zip(all_prompts, all_tokens)
    ])

    # Calculate mean of requested domains
    mean_prompts = text_features.mean(dim=0)
    mean_prompts = mean_prompts / mean_prompts.norm(dim=-1, keepdim=True)
    mean_prompts = mean_prompts.detach().clone()

    # Print debug information
    # print(f"Loaded domains: {doms}")
    # print(f"Learned prompts shape: {text_features.shape}")
    # print(f"Mean prompts shape: {mean_prompts.shape}")

    return mean_prompts




def load_learned_prompts(classnames, num_context_tokens=4, doms=None, device="cuda"):
    
    
    
    plip = CLIPModel.from_pretrained("vinid/plip")
    plip_processor = CLIPProcessor.from_pretrained("vinid/plip")
    
    # Define domain paths in a dictionary
    DOMAIN_PATHS = {
        0: "/home/user/dg/Prompts/output_kg/0/2025-02-18_18-28-09/best_prompt_learner.pth",
        1: "/home/user/dg/Prompts/output_kg/1/2025-02-18_21-11-52/best_prompt_learner.pth",
        2: "/home/user/dg/Prompts/output_kg/2/2025-02-18_18-28-29/best_prompt_learner.pth",
        3: "/home/user/dg/Prompts/output_kg/3/2025-02-18_18-28-35/best_prompt_learner.pth",
        4: "/home/user/dg/Prompts/output_kg/4/2025-02-18_19-57-44/best_prompt_learner.pth"
    }


    # Validate requested domains
    doms = [int(d) for d in doms] if doms else list(DOMAIN_PATHS.keys())
    invalid_doms = [d for d in doms if d not in DOMAIN_PATHS]
    if invalid_doms:
        raise ValueError(f"Invalid domains requested: {invalid_doms}")

    # Load all requested domain prompt learners
    prompt_learners = []
    for domain in doms:
        # Load checkpoint
        checkpoint = torch.load(DOMAIN_PATHS[domain], map_location= device)

        
        # Create and configure prompt learner
        pl = PromptLearner(classnames, plip, plip_processor, num_context_tokens)
        pl.load_state_dict(checkpoint)
        pl.to(device)
        prompt_learners.append(pl)

    # Process all prompts in batch
    text_encoder = TextEncoder(plip).to(device)
    
    # Collect all prompts and tokens
    all_prompts = []
    all_tokens = []
    for pl in prompt_learners:
        with torch.no_grad():
            prompts = pl()
            all_prompts.append(prompts)
            all_tokens.append(pl.tokenized_learnable_prompts)

    # Encode all text features in one batch
    text_features = torch.stack([
        text_encoder(p, t) 
        for p, t in zip(all_prompts, all_tokens)
    ])

    # Calculate mean of requested domains
    mean_prompts = text_features.mean(dim=0)
    mean_prompts = mean_prompts / mean_prompts.norm(dim=-1, keepdim=True)
    mean_prompts = mean_prompts.detach().clone()

    # Print debug information
    # print(f"Loaded domains: {doms}")
    # print(f"Learned prompts shape: {text_features.shape}")
    # print(f"Mean prompts shape: {mean_prompts.shape}")

    return mean_prompts


def load_learned_prompts_val1(classnames, num_context_tokens=4, doms=None, device="cuda"):
    
    # Define domain paths in a dictionary
    DOMAIN_PATHS = {
        0: "/home/user/dg/Prompts/output_kg/diff_val/0/2025-02-19_23-00-54/best_prompt_learner.pth",
        2: "/home/user/dg/Prompts/output_kg/diff_val/2/2025-02-20_01-57-52/best_prompt_learner.pth",
        3: "/home/user/dg/Prompts/output_kg/diff_val/3/2025-02-19_23-00-40/best_prompt_learner.pth",
        4: "/home/user/dg/Prompts/output_kg/diff_val/4/2025-02-19_23-00-38/best_prompt_learner.pth"
    }
    print('------------------------- using val1 prompt loader ----------------------')

    plip = CLIPModel.from_pretrained("vinid/plip")
    plip_processor = CLIPProcessor.from_pretrained("vinid/plip")
    
    # Validate requested domains
    doms = [int(d) for d in doms] if doms else list(DOMAIN_PATHS.keys())
    invalid_doms = [d for d in doms if d not in DOMAIN_PATHS]
    if invalid_doms:
        raise ValueError(f"Invalid domains requested: {invalid_doms}")

    # Load all requested domain prompt learners
    prompt_learners = []
    for domain in doms:
        # Load checkpoint
        checkpoint = torch.load(DOMAIN_PATHS[domain], map_location= device)
        
        # Create and configure prompt learner
        pl = PromptLearner(classnames, plip, plip_processor, num_context_tokens)
        pl.load_state_dict(checkpoint)
        pl.to(device)
        prompt_learners.append(pl)

    # Process all prompts in batch
    text_encoder = TextEncoder(plip).to(device)
    
    # Collect all prompts and tokens
    all_prompts = []
    all_tokens = []
    for pl in prompt_learners:
        with torch.no_grad():
            prompts = pl()
            all_prompts.append(prompts)
            all_tokens.append(pl.tokenized_learnable_prompts)

    # Encode all text features in one batch
    text_features = torch.stack([
        text_encoder(p, t) 
        for p, t in zip(all_prompts, all_tokens)
    ])

    # Calculate mean of requested domains
    mean_prompts = text_features.mean(dim=0)
    mean_prompts = mean_prompts / mean_prompts.norm(dim=-1, keepdim=True)
    mean_prompts = mean_prompts.detach().clone()

    # Print debug information
    # print(f"Loaded domains: {doms}")
    # print(f"Learned prompts shape: {text_features.shape}")
    # print(f"Mean prompts shape: {mean_prompts.shape}")

    return mean_prompts



def load_learned_prompts_kather(classnames, num_context_tokens=4, doms=None, device="cuda"):
    
    # Define domain paths in a dictionary
    DOMAIN_PATHS = {
        0: "/home/user/dg/Prompts/output_kg/kather/0/2025-02-25_21-32-08/best_prompt_learner.pth",
        1: "/home/user/dg/Prompts/output_kg/kather/1/2025-02-25_21-55-12/best_prompt_learner.pth",
        2: "/home/user/dg/Prompts/output_kg/kather/2/2025-02-25_22-12-21/best_prompt_learner.pth",
        3: "/home/user/dg/Prompts/output_kg/kather/3/2025-02-25_22-26-23/best_prompt_learner.pth",
        4: "/home/user/dg/Prompts/output_kg/kather/4/2025-02-25_23-04-49/best_prompt_learner.pth"
    }

    
    plip = CLIPModel.from_pretrained("vinid/plip")
    plip_processor = CLIPProcessor.from_pretrained("vinid/plip")
    
    # Validate requested domains
    doms = [int(d) for d in doms] if doms else list(DOMAIN_PATHS.keys())
    invalid_doms = [d for d in doms if d not in DOMAIN_PATHS]
    if invalid_doms:
        raise ValueError(f"Invalid domains requested: {invalid_doms}")

    # Load all requested domain prompt learners
    prompt_learners = []
    for domain in doms:
        # Load checkpoint
        checkpoint = torch.load(DOMAIN_PATHS[domain], map_location= device)
        
        # Create and configure prompt learner
        pl = PromptLearner(classnames, plip, plip_processor, num_context_tokens)
        pl.load_state_dict(checkpoint)
        pl.to(device)
        prompt_learners.append(pl)

    # Process all prompts in batch
    text_encoder = TextEncoder(plip).to(device)
    
    # Collect all prompts and tokens
    all_prompts = []
    all_tokens = []
    for pl in prompt_learners:
        with torch.no_grad():
            prompts = pl()
            all_prompts.append(prompts)
            all_tokens.append(pl.tokenized_learnable_prompts)

    # Encode all text features in one batch
    text_features = torch.stack([
        text_encoder(p, t) 
        for p, t in zip(all_prompts, all_tokens)
    ])

    # Calculate mean of requested domains
    mean_prompts = text_features.mean(dim=0)
    mean_prompts = mean_prompts / mean_prompts.norm(dim=-1, keepdim=True)
    mean_prompts = mean_prompts.detach().clone()

    # Print debug information
    # print(f"Loaded domains: {doms}")
    # print(f"Learned prompts shape: {text_features.shape}")
    # print(f"Mean prompts shape: {mean_prompts.shape}")

    return mean_prompts



# def load_learned_prompts_mean(doms, classnames, num_context_tokens=4, device="cuda"):
    
#     file_name = '_'.join(map(str, doms))
    
#     plip = CLIPModel.from_pretrained("vinid/plip")
#     plip_processor = CLIPProcessor.from_pretrained("vinid/plip")
    
#     # Define domain paths in a dictionary
#     DOMAIN_PATHS = {
#         "1_2_3": "/home/user/dg/Prompts/output_kg/1_2_3/2025-02-17_21-03-20/last_prompt_learner.pth",
#         "1_2_4": "/home/user/dg/Prompts/output_kg/1_2_4/2025-02-17_21-03-32/last_prompt_learner.pth",
#         "1_3_4": "/home/user/dg/Prompts/output_kg/1_3_4/2025-02-17_21-03-44/last_prompt_learner.pth",
#         "2_3_4": "/home/user/dg/Prompts/output_kg/2_3_4/2025-02-17_21-03-54/last_prompt_learner.pth",
#     }
    

#     # Validate requested domains
#     # doms = [int(d) for d in doms] if doms else list(DOMAIN_PATHS.keys())
#     # invalid_doms = [d for d in doms if d not in DOMAIN_PATHS]
#     # if invalid_doms:
#     #     raise ValueError(f"Invalid domains requested: {invalid_doms}")

#     # Load all requested domain prompt learners
#     prompt_learners = []
#     # for domain in doms:
#         # Load checkpoint
#     checkpoint = torch.load(DOMAIN_PATHS[file_name], map_location= device)

        
#     # Create and configure prompt learner
#     pl = PromptLearner(classnames, plip, plip_processor, num_context_tokens)
#     pl.load_state_dict(checkpoint)
#     pl.to(device)
#     prompt_learners.append(pl)

#     # Process all prompts in batch
#     text_encoder = TextEncoder(plip).to(device)
    
#     # Collect all prompts and tokens
#     all_prompts = []
#     all_tokens = []
#     for pl in prompt_learners:
#         with torch.no_grad():
#             prompts = pl()
#             all_prompts.append(prompts)
#             all_tokens.append(pl.tokenized_learnable_prompts)

#     # Encode all text features in one batch
#     text_features = torch.stack([
#         text_encoder(p, t) 
#         for p, t in zip(all_prompts, all_tokens)
#     ])

#     # Calculate mean of requested domains
#     mean_prompts = text_features.mean(dim=0)
#     mean_prompts = mean_prompts / mean_prompts.norm(dim=-1, keepdim=True)
#     mean_prompts = mean_prompts.detach().clone()

#     # Print debug information
#     # print(f"Loaded domains: {doms}")
#     # print(f"Learned prompts shape: {text_features.shape}")
#     # print(f"Mean prompts shape: {mean_prompts.shape}")
#     "==================== prompts loaded successfully ====================="

#     return mean_prompts
