from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch

import pandas as pd
from tqdm import tqdm

from PIL import Image
from transformers import CLIPProcessor, CLIPModel

import os
import sys

sys.path.append(os.path.abspath('/home/user/dg'))
sys.path.append(os.path.abspath('/home/user/dg/Prompt/KG/KgCoOp/OurUtils'))
sys.path.append(os.path.abspath('/home/user/dg/utils'))
sys.path.append(os.path.abspath('/home/user/dg/Prompts'))
from avg_template_prompts import get_text_feature
from datasets import MultipleDomainDataset

# Define root directory and domains
root_dir = '/home/user/data/camelyon17dc'
domains = ['0', '1', '2', '3', '4']

# Define transformations (modify as needed)
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Create datasets and dataloaders
datasets = {domain: MultipleDomainDataset(root_dir, [domain], transform=transform) for domain in domains}
dataloaders = {domain: DataLoader(datasets[domain], batch_size=128, shuffle=True,pin_memory=True , num_workers=4) for domain in domains}


plip = CLIPModel.from_pretrained("vinid/plip")
plip_processor = CLIPProcessor.from_pretrained("vinid/plip")


gpu_index = 4
if torch.cuda.is_available() and gpu_index < torch.cuda.device_count():
        device = torch.device(f"cuda:{gpu_index}")
        print(f"Using device: {device} - {torch.cuda.get_device_name(gpu_index)}")
else:
        device = torch.device("cpu")
        print("Using CPU")
        
plip.to(device)


def calculate_metrics(predictions, true_labels):
    """Calculate accuracy, precision, recall, and F1 score."""
    tp = ((predictions == 1) & (true_labels == 1)).sum().item()
    tn = ((predictions == 0) & (true_labels == 0)).sum().item()
    fp = ((predictions == 1) & (true_labels == 0)).sum().item()
    fn = ((predictions == 0) & (true_labels == 1)).sum().item()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0

    return accuracy, precision, recall, f1_score

# Loop through all combinations of texts



def zero_shot_classification(loader,text_features,processor,model):
    predictions = []
    true_labels = []
    

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Processing", unit="batch")):
            images, labels = batch[0].to(device), batch[1].to(device)

            # Convert images to PIL format (if not already)
            pil_images = [transforms.ToPILImage()(img) for img in images]

            # Tokenize text and image inputs
            inputs = processor(images=pil_images, return_tensors="pt", padding=True)

            # Move inputs to the same device as model
            inputs_images = {key: val.to(device) for key, val in inputs.items()}

            # Get model outputs
            CLIP_image_features = model.get_image_features(**inputs_images)
            CLIP_image_features /= CLIP_image_features.norm(dim=-1, keepdim=True)
            clip_logits = (100.0 * CLIP_image_features @ text_features.T).type(torch.float32)

            # Extract similarity scores
            probs = clip_logits.softmax(dim=1)  # Convert to probabilities

            # Get predicted class (index with max probability)
            preds = probs.argmax(dim=1).cpu()

            # Store results
            predictions.extend(preds.numpy())
            true_labels.extend(labels.cpu().numpy())

    # Convert results to tensors
    predictions = torch.tensor(predictions)
    true_labels = torch.tensor(true_labels)

    # Calculate metrics
    accuracy, precision, recall, f1_score = calculate_metrics(predictions, true_labels)
    
    return accuracy, precision, recall, f1_score


agg_text_feature_template = get_text_feature(plip_processor, plip, device)
agg_text_feature_template /= agg_text_feature_template.norm(dim=-1, keepdim=True)
agg_text_feature_template = agg_text_feature_template.to(device)



# Iterate through each domain and compute metrics
results_agg = {}

print("aggregated results")
for domain, loader in dataloaders.items():
    print(f"\nEvaluating zero-shot classification for domain: {domain}")
    
    accuracy, precision, recall, f1_score = zero_shot_classification(loader, agg_text_feature_template, plip_processor, plip)
    print(f'accuracy: {accuracy}')
    print(f'precision: {precision}')
    print(f'recall: {recall}')
    print(f'f1_score: {f1_score}')

    
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


# Generate text features for classification
text_classes = ["a patch of normal tissue", "a patch of cancerous tissue"]
text_inputs = plip_processor(text=text_classes, return_tensors="pt", padding=True).to(device)
text_features = plip.get_text_features(**text_inputs)
text_features /= text_features.norm(dim=-1, keepdim=True)


# Iterate through each domain and compute metrics
results_hand = {}

for domain, loader in dataloaders.items():
    print(f"\nEvaluating zero-shot classification for domain: {domain}")
    
    accuracy, precision, recall, f1_score = zero_shot_classification(loader, text_features, plip_processor, plip)
    
    results_hand[domain] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1_score
    }


# Print the results in a readable format
print("\n===== Zero-Shot Classification Results handcrafted =====")
all_accuracy = []
all_precision = []
all_recall = []
all_f1 = []

for domain, metrics in results_hand.items():
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
