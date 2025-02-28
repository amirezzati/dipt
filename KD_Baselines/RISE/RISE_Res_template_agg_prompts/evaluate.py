# main
import sys
import os
from pathlib import Path
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from tqdm import tqdm
from PIL import Image
import argparse
import json


sys.path.append(os.path.abspath('/home/user/dg/'))
sys.path.append(os.path.abspath('/home/user/dg/Prompts'))
sys.path.append(os.path.abspath('/home/user/dg/utils'))
sys.path.append(os.path.abspath('/home/user/dg/RISE'))

from datasets import MultipleDomainDataset
from rise_utils import do_test_res
from  RISE.timm.models import create_model

def get_test_domain(model_name):
    """
    Extracts all numbers from the model name (both before and after 'v') 
    and returns the complement of the full domain set {0,1,2,3,4}.
    """
    # Extract all numbers from the entire model name
    numbers = []
    for part in model_name.split("_"):
        if part.isdigit():
            numbers.append(int(part))
        # Handle "v" separator (e.g., in "0_1_2v_3")
        elif 'v' in part:
            for subpart in part.split('v'):
                if subpart.isdigit():
                    numbers.append(int(subpart))

    # Full set of possible domains
    full_set = {0, 1, 2, 3, 4,5,6}
    
    # Compute the complement
    complement = full_set - set(numbers)
    
    # Convert to sorted list of strings
    return [str(x) for x in sorted(complement)]

# Rest of the code remains the same as in previous answer

def main():
    parser = argparse.ArgumentParser(description="Evaluate resnet student of rise.")
    parser.add_argument('--models', nargs='+', required=True, 
                      help='List of paths to model files')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save results')
    parser.add_argument('--gpu_index', type=int, required=True,
                      help="GPU index to use")
    
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Device setup
    device = torch.device(f"cuda:{args.gpu_index}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Using GPU {args.gpu_index}: {torch.cuda.get_device_name(args.gpu_index)}")
    else:
        print("Using CPU")



    # Process each model
    for model_path in args.models:
        try:
            print(f"\n{'='*40}")
            print(f"Processing model: {Path(model_path).name}")
            print(f"{'='*40}")
            
            # Get the test domain from the model name
            model_name = Path(model_path).name
            # test_domain = get_test_domain(model_name)
            test_domain = ['6']
            print(f"Test domain: {test_domain}")
            
            # Prepare test data for the test domain
            root_dir = '/home/user/data/kather100k/train'

            transform = transforms.Compose([transforms.ToTensor()])
            test_data = MultipleDomainDataset(root_dir, test_domain, transform)
            test_loader = DataLoader(test_data, batch_size=128, shuffle=False,pin_memory = True , num_workers = 4)
            
            # Load model
            student = create_model("resnetv2_50x1_bit.goog_in21k_ft_in1k", pretrained=True, num_classes=9)
            student.to(device)


            
            student.load_state_dict(torch.load(model_path))

            # Evaluate
            accuracy, precision, recall, f1_score = do_test_res(test_loader, student, device)

            # Save results
            results_file = Path(args.output_dir) / f"{Path(model_path).stem}_results.txt"
            with open(results_file, 'w') as f:
                f.write(f"Model: {model_path}\n")
                f.write(f"Test Domain: {test_domain}\n")
                f.write(f"Accuracy: {accuracy:.4f}\n")
                f.write(f"Precision: {precision:.4f}\n")
                f.write(f"Recall: {recall:.4f}\n")
                f.write(f"F1 Score: {f1_score:.4f}\n")

            print(f"Results saved to {results_file}")
            
            # Clean up to free GPU memory
            del student
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error processing {model_path}: {str(e)}")
            continue

if __name__ == "__main__":
    main()