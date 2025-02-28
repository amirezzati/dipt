import os
import sys
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from transformers import CLIPProcessor, CLIPModel
import argparse

sys.path.append(os.path.abspath('/home/user/dg'))
sys.path.append(os.path.abspath('/home/user/dg/Prompt/KG/KgCoOp/OurUtils'))
sys.path.append(os.path.abspath('/home/user/dg/utils'))
sys.path.append(os.path.abspath('/home/user/dg/Prompts'))

from datasets import MultipleDomainDataset
from avg_template_prompts import get_text_feature
from prompts_utils import zero_shot_classification, TextEncoder, PromptLearner

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run zero-shot classification with specified parameters.")
    parser.add_argument("--gpu_index", type=int, default=4, help="Index of the GPU to use (default: 4).")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the checkpoint file.")
    parser.add_argument("--num_context_tokens", type=int, default=4, help="Number of context tokens (default: 4).")
    args = parser.parse_args()

    # Define root directory and domains
    root_dir = '/home/user/data/kather100k/train'
    domains = ['6', '4']

    # Define transformations (modify as needed)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Create datasets and dataloaders
    datasets = {domain: MultipleDomainDataset(root_dir, [domain], transform=transform) for domain in domains}
    dataloaders = {domain: DataLoader(datasets[domain], batch_size=128, shuffle=False,pin_memory = True,num_workers=4) for domain in domains}

    # Device configuration
    if torch.cuda.is_available() and args.gpu_index < torch.cuda.device_count():
        device = torch.device(f"cuda:{args.gpu_index}")
        print(f"Using device: {device} - {torch.cuda.get_device_name(args.gpu_index)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Load PLIP model and processor
    plip = CLIPModel.from_pretrained("vinid/plip").to(device)
    plip_processor = CLIPProcessor.from_pretrained("vinid/plip")

    # Load PromptLearner
    classnames = ["normal", "tumor","tumor","tumor","tumor","tumor","tumor","tumor","tumor"]
    print(f'n_ctx = {args.num_context_tokens} ')
    prompt_learner = PromptLearner(classnames, plip, plip_processor, args.num_context_tokens)
    checkpoint = torch.load(args.checkpoint_path)
    prompt_learner.load_state_dict(checkpoint)
    prompt_learner.to(device)
    print("PromptLearner successfully loaded!")

    # Generate prompts and text features
    prompts = prompt_learner()
    tokenized_prompts = prompt_learner.tokenized_learnable_prompts
    text_encoder = TextEncoder(plip)
    text_features_prompts = text_encoder(prompts, tokenized_prompts)
    text_features_prompts /= text_features_prompts.norm(dim=-1, keepdim=True)

    print("-" * 50)
    print("Text features size:", text_features_prompts.size())
    print("Text features:", text_features_prompts)

    # Iterate through each domain and compute metrics
    results_agg = {}
    print("Aggregated results:")
    for domain, loader in dataloaders.items():
        print(f"\nEvaluating zero-shot classification for domain: {domain}")
        accuracy, precision, recall, f1_score = zero_shot_classification(
            loader, text_features_prompts, plip_processor, plip, device
        )
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1_score}")

        results_agg[domain] = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1_score
        }

    # Print the results in a readable format
    print("\n===== Zero-Shot Classification Results =====")
    all_accuracy = []
    all_precision = []
    all_recall = []
    all_f1 = []

    for domain, metrics in results_agg.items():
        print(f"\n📌 Domain {domain}:")
        print(f"   ✅ Accuracy  : {metrics['Accuracy']:.4f}")
        print(f"   🎯 Precision : {metrics['Precision']:.4f}")
        print(f"   🔁 Recall    : {metrics['Recall']:.4f}")
        print(f"   📊 F1 Score  : {metrics['F1 Score']:.4f}")

        all_accuracy.append(metrics['Accuracy'])
        all_precision.append(metrics['Precision'])
        all_recall.append(metrics['Recall'])
        all_f1.append(metrics['F1 Score'])

    # Compute the average for each metric
    avg_accuracy = sum(all_accuracy) / len(all_accuracy)
    avg_precision = sum(all_precision) / len(all_precision)
    avg_recall = sum(all_recall) / len(all_recall)
    avg_f1 = sum(all_f1) / len(all_f1)

    print("\n===== Average Metrics Across All Domains =====")
    print(f"   ✅ Average Accuracy  : {avg_accuracy:.4f}")
    print(f"   🎯 Average Precision : {avg_precision:.4f}")
    print(f"   🔁 Average Recall    : {avg_recall:.4f}")
    print(f"   📊 Average F1 Score  : {avg_f1:.4f}")

    # Save results to a text file
    results_dir = os.path.dirname(args.checkpoint_path)
    checkpoint_type = os.path.basename(args.checkpoint_path).split('_')[0]  # 'best' or 'last'
    results_file = os.path.join(results_dir, f"results_{checkpoint_type}.txt")
    with open(results_file, "w") as f:
        f.write("===== Zero-Shot Classification Results =====\n")
        for domain, metrics in results_agg.items():
            f.write(f"\n📌 Domain {domain}:\n")
            f.write(f"   ✅ Accuracy  : {metrics['Accuracy']:.4f}\n")
            f.write(f"   🎯 Precision : {metrics['Precision']:.4f}\n")
            f.write(f"   🔁 Recall    : {metrics['Recall']:.4f}\n")
            f.write(f"   📊 F1 Score  : {metrics['F1 Score']:.4f}\n")

        f.write("\n===== Average Metrics Across All Domains =====\n")
        f.write(f"   ✅ Average Accuracy  : {avg_accuracy:.4f}\n")
        f.write(f"   🎯 Average Precision : {avg_precision:.4f}\n")
        f.write(f"   🔁 Average Recall    : {avg_recall:.4f}\n")
        f.write(f"   📊 Average F1 Score  : {avg_f1:.4f}\n")

    print(f"\nResults saved to: {results_file}")

if __name__ == "__main__":
    main()