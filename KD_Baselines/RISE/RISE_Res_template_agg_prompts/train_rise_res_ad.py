import sys
import os

import torch
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
import torchvision.models as models


from transformers import CLIPProcessor, CLIPModel


sys.path.append(os.path.abspath('/home/user/dg/'))
sys.path.append(os.path.abspath('/home/user/dg/Prompts'))
sys.path.append(os.path.abspath('/home/user/dg/utils'))
sys.path.append(os.path.abspath('/home/user/dg/RISE'))


from datasets import MultipleDomainDataset
from rise_utils import get_optim_and_scheduler  , do_training_plip_res_AD
from ViTWithProjectionAndClassifier import ViTWithProjectionAndClassifier
from load_prompts import get_learned_prompts
from  RISE.timm.models import create_model
from avg_template_prompts import get_text_feature_kather



import argparse



def train_all_rise(epochs, train_doms, batch_size,
                   clip, clip_processor,
                   classnames, 
                   distill_weight, classification_weight, distance_weight, T,
                   device, learning_rate,  
                   eval_interval=200,
                   model_save_path="/home/user01/dg/RISE/train_output/CAM17"):
    
    # Define all possible domains
    all_domains = {'0', '1', '2', '3', '4'}
    transform = transforms.Compose([transforms.ToTensor()])



    # Iterate through each training domain combination
    for train_domains in train_doms:
        # Calculate validation domains as complement
        train_set = set(train_domains)
        val_domains = ['5']
        
        print(f'\nTraining domains: {train_domains}')
        print(f'Validation domains: {val_domains}')

        # Calculate mean_prompts for current training domains
        agg_text_feature_template = get_text_feature_kather(clip_processor, clip, device)
        agg_text_feature_template /= agg_text_feature_template.norm(dim=-1, keepdim=True)
        agg_text_feature_template = agg_text_feature_template.to(device)
            
        # Initialize student model

        
        student = create_model("resnetv2_50x1_bit.goog_in21k_ft_in1k", pretrained=True, num_classes=len(classnames))
        
        student.fc.weight.datadata = agg_text_feature_template.clone()
        student.to(device)

        
        # Create optimizer/scheduler
        optimizer, scheduler = get_optim_and_scheduler(
            student, epochs, learning_rate, True, False
        )
        
        root_dir = '/home/user/data/kather100k/train'



        # Prepare datasets
        train_dataset = MultipleDomainDataset(
            root_dir=root_dir,
            domains=train_domains,
            transform=transform
        )
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=4, 
            pin_memory=True
        )

        # Prepare validation loaders
        val_loaders = {
            domain: DataLoader(
                MultipleDomainDataset(
                    root_dir=root_dir,
                    domains=[domain],
                    transform=transform
                ),
                batch_size=128,
                shuffle=True
            ) for domain in val_domains
        }


        # Run training
        do_training_plip_res_AD(
            epochs=epochs,
            train_loader=train_loader,
            val_loader=val_loaders,
            doms={'train_doms': train_domains, 'val_doms': val_domains},
            student=student,
            clip=clip,
            clip_processor=clip_processor,
            text_features_ems=agg_text_feature_template,
            distill_weight=distill_weight,
            classification_weight=classification_weight,
            distance_weight=distance_weight,
            T=T,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            eval_interval=eval_interval,
            model_save_path=model_save_path
        )
        
        del student, optimizer, scheduler
        torch.cuda.empty_cache()  # Then actually free memory



def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train Domain Generalization Model")
    
    # Required arguments
    parser.add_argument('--train_doms', nargs='+', required=True,
                      help='List of training domain groups (e.g., "0,1,2" "0,3,4")')
    parser.add_argument('--classnames', nargs='+', required=True,
                      default=["normal", "tumor"],
                      help='List of class names (default: ["normal", "tumor"])')
    
    # Optional arguments
    parser.add_argument('--gpu_index', type=int, default=4,
                      help='GPU index to use (default: 4)')
    parser.add_argument('--epochs', type=int, default=1,
                      help='Number of training epochs (default: 1)')
    parser.add_argument('--batch_size', type=int, default=128,
                      help='Batch size for training (default: 128)')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                      help='Learning rate (default: 0.001)')
    parser.add_argument('--distill_weight', type=float, default=0.3,
                      help='Distillation loss weight (default: 0.3)')
    parser.add_argument('--classification_weight', type=float, default=0.4,
                      help='Classification loss weight (default: 0.4)')
    parser.add_argument('--distance_weight', type=float, default=0.3,
                      help='Distance loss weight (default: 0.3)')
    parser.add_argument('--T', type=float, default=2.0,
                      help='Temperature parameter (default: 2.0)')
    parser.add_argument('--model_save_path', 
                      default="/home/user/dg/RISE/RISE_Res_learnable_prompts/Outputs/Models",
                      help='Path to save trained models')

    args = parser.parse_args()
    
    
    # Print all arguments
    print("\n" + "="*40)
    print("Configuration Parameters:")
    print("="*40)
    max_len = max(len(arg) for arg in vars(args))
    for arg in sorted(vars(args)):
        value = getattr(args, arg)
        if isinstance(value, list):
            # Format list parameters more readably
            formatted_value = "\n  " + "\n  ".join(str(item) for item in value)
        else:
            formatted_value = str(value)
        print(f"{arg:<{max_len}} : {formatted_value}")
    print("="*40 + "\n")

    # Device configuration
    device = torch.device(f"cuda:{args.gpu_index}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load CLIP models
    plip = CLIPModel.from_pretrained("vinid/plip").to(device)
    plip_processor = CLIPProcessor.from_pretrained("vinid/plip")

    # Process training domains
    train_doms = [group.split(',') for group in args.train_doms]

    # Training configuration
    config = {
        'classnames': args.classnames,  # Now using parsed classnames
        'train_doms': train_doms,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'distill_weight': args.distill_weight,
        'classification_weight': args.classification_weight,
        'distance_weight': args.distance_weight,
        'T': args.T,
    }

    # Start training process
    train_all_rise(
            **config,
            clip=plip,
            clip_processor=plip_processor,
            device=device,
            eval_interval=200,
            model_save_path=args.model_save_path
        )
    
    # Cleanup
    del plip, plip_processor
    torch.cuda.empty_cache()  # Then actually free memory


if __name__ == "__main__":
    # Example usage:
    # python script.py --train_doms 0,1,2 0,3,4 --classnames normal tumor
    main()