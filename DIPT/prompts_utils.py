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



sys.path.append(os.path.abspath('/home/user/dg/Prompts'))




class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        """
        A text encoder module that processes text prompts using a CLIP-based model.
        
        Args:
            clip_model: A pre-trained CLIP model containing the text encoder.
        """
        super().__init__()
        self.clip_model = clip_model  # Store the CLIP model
        self.dtype = clip_model.dtype  # Store the model's  data type

    def forward(self, embedded_tokens, tokenized_prompts):
        """
        Forward pass of the text encoder.
        
        Args:
            embedded_tokens (torch.Tensor): Input token embeddings of shape (batch_size, seq_len, hidden_dim).
            tokenized_prompts (torch.Tensor): Tokenized input prompts, used to extract the CLS token.
        
        Returns:
            torch.Tensor: Pooled output representation of shape (batch_size, hidden_dim).
        """
        # Add position embeddings to the input token embeddings
        position_embeddings = self.clip_model.text_model.embeddings.position_embedding.weight.unsqueeze(0)
        embedded_tokens += position_embeddings[:, :embedded_tokens.shape[1], :]

        # Pass the embeddings through the text encoder
        encoder_outputs = self.clip_model.text_model.encoder(inputs_embeds=embedded_tokens)

        # Extract the last hidden state from encoder outputs
        last_hidden_state = encoder_outputs[0]

        # Apply the final layer normalization
        normalized_hidden_state = self.clip_model.text_model.final_layer_norm(last_hidden_state)

        # Extract the pooled output using the position of the highest-value token in each sequence
        batch_indices = torch.arange(normalized_hidden_state.shape[0], device=tokenized_prompts.device)
        pooled_output = normalized_hidden_state[batch_indices, tokenized_prompts.to(torch.int).argmax(dim=-1)]

        return pooled_output


class PromptLearner(nn.Module):
    def __init__(self, class_names, clip_model, processor, num_context_tokens, agg_vector=None):
        super().__init__()
        self.class_names = class_names
        self.num_context_tokens = num_context_tokens

        self.num_classes = len(class_names)
        self.embedding_dim = 512  
        self.dtype = clip_model.dtype  

        # Initialize agg_vector if None
        if agg_vector is None:
            print("Initializing agg_vector as a tensor of zeros.")
            agg_vector = torch.zeros((self.num_classes, self.embedding_dim), dtype=self.dtype, device=clip_model.device)
        self.agg_vector = agg_vector  # Store the aggregate vector

        # Initialize context vectors
        context_vectors = self.initialize_context_vectors(clip_model)
        self.context_tokens = nn.Parameter(context_vectors)

        # Generate prompts
        self.prompt_prefix = " ".join(["X"] * self.num_context_tokens)

        self.text_features_base = agg_vector

        # Tokenize full promptsprompts using the appropriate method
        print("Initializing PromptLearner with agg_vector.")
        token_embeddings = self.tokenize_full_prompts_agg(processor , clip_model,self.agg_vector)

        # Register prefix and suffix tokens
        self.register_buffer("prefix_tokens", token_embeddings[:, :1, :].to(clip_model.device))
        self.register_buffer("suffix_tokens", token_embeddings[:, 1 + self.num_context_tokens:, :].to(clip_model.device))

    def initialize_context_vectors(self,clip_model):
        print("Initializing domain-specific contexts")
        context_shape = (self.num_classes, self.num_context_tokens, self.embedding_dim)
        context_vectors = torch.empty(context_shape, dtype=self.dtype, device=clip_model.device)
        nn.init.normal_(context_vectors, std=0.02)
        return context_vectors

    def tokenize_full_prompts_agg(self,processor , clip_model ,agg_vector):
        full_prompts = [self.prompt_prefix + " C " for _ in range(len(self.class_names))]

        # Tokenize prompts
        self.tokenized_learnable_prompts = processor(
            text=full_prompts, return_tensors="pt", padding=True
        ).input_ids.to(clip_model.device)

        with torch.no_grad():
            token_embeddings = clip_model.text_model.embeddings.token_embedding(
                self.tokenized_learnable_prompts
            ).type(self.dtype)
        print("Tokenized full prompts shape (with aggregate vector):", token_embeddings.shape)
        # Apply agg_vector modification
        token_embeddings[:, 1 + self.num_context_tokens, :] = agg_vector
        return token_embeddings

    def forward(self):
        context_tokens = self.context_tokens
        if context_tokens.dim() == 2:
            context_tokens = context_tokens.unsqueeze(0).expand(self.num_classes, -1, -1)
        # Ensure context_tokens is on the same device as prefix_tokens and suffix_tokens
        context_tokens = context_tokens.to(self.prefix_tokens.device)
        return torch.cat([self.prefix_tokens, context_tokens, self.suffix_tokens], dim=1)


class PromptLearnerLearnableAgg(PromptLearner):
    def __init__(self, class_names, clip_model, processor, num_context_tokens, agg_vector=None):
        # Call the parent initializer first.
        super().__init__(class_names, clip_model, processor, num_context_tokens, agg_vector=agg_vector)
        # Convert the agg_vector to a learnable parameter.
        self.agg_vector = nn.Parameter(self.agg_vector.detach().clone())
        # Store processor and clip_model for reusing in forward (if not already stored)
        
        # Recompute token embeddings with the current agg_vector value.
        token_embeddings = self.tokenize_full_prompts_agg(processor, clip_model, self.agg_vector)
        # Instead of using the fixed registered buffers, extract prefix and suffix tokens on the fly.
        self.register_buffer("prefix_tokens", token_embeddings[:, :1, :].to(clip_model.device))
        self.register_buffer("suffix_tokens", token_embeddings[:, 1 + self.num_context_tokens:, :].to(clip_model.device))





#######################################
# CustomCLIP Class
#######################################
# class CustomCLIP(nn.Module):
#     def __init__(self, classnames, clip_model, processor,num_context_tokens=4, agg_vector=None):
#         super().__init__()
#         self.prompt_learner = PromptLearner(classnames, clip_model, processor, num_context_tokens, agg_vector=agg_vector)
#         self.tokenized_prompts = self.prompt_learner.tokenized_learnable_prompts
#         self.ori_embedding = self.prompt_learner.text_features_base
#         self.text_encoder = TextEncoder(clip_model)
#         self.logit_scale = clip_model.logit_scale
#         self.dtype = clip_model.dtype

#         self.clip_model = clip_model
#         self.vision_model = clip_model.vision_model
#         self.visual_projection = clip_model.visual_projection
#         self.processor = processor

#     def forward(self, image):
#         pil_images = [transforms.ToPILImage()(img) for img in image]
#         inputs = self.processor(images=pil_images, return_tensors="pt", padding=True)
#         pixel_values = inputs.pixel_values.to(self.logit_scale.device)
#         prompts = self.prompt_learner()
#         vision_outputs = self.vision_model(pixel_values=pixel_values.type(self.dtype))
#         image_embeds = vision_outputs.pooler_output
#         image_features = self.visual_projection(image_embeds)
#         text_features = self.text_encoder(prompts, self.tokenized_prompts)
#         text_features_old = self.ori_embedding

#         image_features = image_features / image_features.norm(dim=-1, keepdim=True)
#         text_features = text_features / text_features.norm(dim=-1, keepdim=True)
#         logit_scale = self.logit_scale.exp()
#         logits = logit_scale * image_features @ text_features.t()

#         cos = nn.CosineSimilarity(dim=1)
#         text_features_old = text_features_old / text_features_old.norm(dim=-1, keepdim=True)
#         score = 1.0 - torch.mean(cos(# The above code seems to be a comment in a Python script. The
#         # text "text_features" is likely a placeholder or a label for a
#         # section of code related to text processing or analysis.
#         # Comments in Python are denoted by the "#" symbol and are used
#         # to provide explanations or notes within the code for better
#         # understanding.
#         text_features, text_features_old))
#         return logits, score
    
    

class CustomCLIP(nn.Module):
    def __init__(self, classnames, clip_model, processor, num_context_tokens=4, agg_vector=None, use_learnable_agg=False):
        super().__init__()
        # Choose the prompt learner variant based on use_learnable_agg flag.
        if use_learnable_agg:
            self.prompt_learner = PromptLearnerLearnableAgg(classnames, clip_model, processor, num_context_tokens, agg_vector=agg_vector)
        else:
            self.prompt_learner = PromptLearner(classnames, clip_model, processor, num_context_tokens, agg_vector=agg_vector)
            
        self.tokenized_prompts = self.prompt_learner.tokenized_learnable_prompts
        self.ori_embedding = self.prompt_learner.text_features_base
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.clip_model = clip_model
        self.vision_model = clip_model.vision_model
        self.visual_projection = clip_model.visual_projection
        self.processor = processor

    def forward(self, image):
        pil_images = [transforms.ToPILImage()(img) for img in image]
        inputs = self.processor(images=pil_images, return_tensors="pt", padding=True)
        pixel_values = inputs.pixel_values.to(self.logit_scale.device)
        prompts = self.prompt_learner()
        vision_outputs = self.vision_model(pixel_values=pixel_values.type(self.dtype))
        image_embeds = vision_outputs.pooler_output
        image_features = self.visual_projection(image_embeds)
        text_features = self.text_encoder(prompts, self.tokenized_prompts)
        text_features_old = self.ori_embedding

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        cos = nn.CosineSimilarity(dim=1)
        text_features_old = text_features_old / text_features_old.norm(dim=-1, keepdim=True)
        score = 1.0 - torch.mean(cos(text_features, text_features_old))
        return logits, score




# def calculate_metrics(predictions, true_labels):
#     """Calculate accuracy, precision, recall, and F1 score.
    
#     Args:
#         predictions (torch.Tensor): Predicted labels (assumed binary: 0 or 1).
#         true_labels (torch.Tensor): True labels (assumed binary: 0 or 1).
    
#     Returns:
#         Tuple[float, float, float, float]: accuracy, precision, recall, f1_score.
#     """
#     tp = ((predictions == 1) & (true_labels == 1)).sum().item()
#     tn = ((predictions == 0) & (true_labels == 0)).sum().item()
#     fp = ((predictions == 1) & (true_labels == 0)).sum().item()
#     fn = ((predictions == 0) & (true_labels == 1)).sum().item()
#     accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
#     precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
#     recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
#     f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
#     return accuracy, precision, recall, f1_score


def calculate_metrics(predictions, true_labels):
    """Calculate accuracy, precision, recall, and F1 score for multi-class classification."""
    
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1_score, _ = precision_recall_fscore_support(true_labels, predictions, average="weighted")

    return accuracy, precision, recall, f1_score

# Loop through all combinations of texts



#######################################
# PromptLearningTrainer Class
#######################################

class PromptLearningTrainer:
    def __init__(self, model, train_loader, val_loader, device, domain_num=None, 
                patient_id=None,  # Add patient ID tracking
                num_epochs=10, lr=1e-3, score_weight=1.0, eval_interval=50, prefix=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.score_weight = score_weight
        
        
        self.domain_num = domain_num
        
        self.eval_interval = eval_interval  # Add this line

        # Add patient ID to output directory
        if patient_id is not None:
            self.patient_id = patient_id
            if prefix:
                prefix = f"{prefix}_patient{patient_id}"
            else:
                prefix = f"patient{patient_id}"
              
        self.prefix = prefix  # <--- ADD THIS LINE
  
                
        # Freeze all parameters except the prompt learner's context tokens.
        # for name, param in self.model.named_parameters():
        #     if "prompt_learner.context_tokens" not in name:
        #         param.requires_grad = False
        #     else:
        #         print(f"Trainable parameter: {name}")
        
# Freeze all parameters except those in the prompt learner that should be trainable.
        for name, param in self.model.named_parameters():
            # If using the learnable agg version, unfreeze both context_tokens and agg_vector.
            if ("prompt_learner.context_tokens" in name or 
                ("prompt_learner.agg_vector" in name and isinstance(self.model.prompt_learner, PromptLearnerLearnableAgg))):
                param.requires_grad = True
                print(f"Trainable parameter: {name}")
            else:
                param.requires_grad = False

        self.optimizer = optim.Adam(self.model.prompt_learner.parameters(), lr=lr)
        self.best_val_acc = 0.0
        self.best_model_state = None

        # Create an output directory based on current date/time
        if domain_num:
            if isinstance(domain_num, (list, tuple)):
                domain_str = '_'.join(map(str, domain_num))
            else:
                domain_str = str(domain_num)
                
                
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


        if self.prefix:
            self.output_dir = os.path.join(
                "output_kg", 
                prefix or "", 
                # f"domain_{'_'.join(map(str, domain_num))}",
                timestamp
            )
        else:
            self.output_dir = os.path.join("output_kg", domain_str, timestamp)
        os.makedirs(self.output_dir, exist_ok=True)
        
        
        self.save_setup_info()  



    def save_setup_info(self):
        """Saves training setup information to a text file in the output directory."""
        setup_info = {
            "Class Names": self.model.prompt_learner.class_names,
            "Number of Context Tokens": self.model.prompt_learner.num_context_tokens,
            "Embedding Dimension": self.model.prompt_learner.embedding_dim,
            "Score Weight": self.score_weight,
            "Evaluation Interval": self.eval_interval,  # Add this line
            "Learning Rate": self.optimizer.param_groups[0]['lr'],
            "Using Agg Vector": self.model.prompt_learner.agg_vector is not None,
            "Domain Number": self.domain_num,
            "Number of Epochs": self.num_epochs,
            "Batch Size": self.train_loader.batch_size,  # Add this line
            "Training Device": str(self.device),
            "Model Type": "CustomCLIP with Prompt Learning"
        }

        file_path = os.path.join(self.output_dir, "training_setup.txt")
        with open(file_path, 'w') as f:
            f.write("Training Setup Information\n")
            f.write("==========================\n")
            for key, value in setup_info.items():
                f.write(f"{key}: {value}\n")
            f.write("\nSaved at: {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))


    def print_setup(self):
        """
        Prints the training setup.
        """
        print("===================================")
        print("Training Setup:")
        print(f"  Num Classes: {self.model.prompt_learner.num_classes}")
        print(f"  Num Context Tokens: {self.model.prompt_learner.num_context_tokens}")
        print(f"  Embedding Dim: {self.model.prompt_learner.embedding_dim}")
        print(f"  Score Weight: {self.score_weight}")
        print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']}")
        if self.model.prompt_learner.agg_vector is not None:
            print("  Running training with agg_vector.")
        else:
            print("  Running training without agg_vector.")

        domain_number = self.domain_num


        print(f"Domain Number: {domain_number}")
        print("Output directory:", self.output_dir)
        print("===================================")

    def parse_batch(self, batch):
        """
        Moves a batch of images and labels to the correct device.
        """
        images, labels = batch[0].to(self.device), batch[1].to(self.device)
        return images, labels

    def forward_backward(self, images, labels, batch_count, epoch):
        """
        Performs a forward and backward pass.
        """
        logits, score = self.model(images)
        loss_ce = F.cross_entropy(logits, labels)
        loss = loss_ce + self.score_weight * score
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        preds = logits.argmax(dim=1).cpu()
        true_labels = labels.cpu()
        batch_acc, batch_prec, batch_rec, batch_f1 = calculate_metrics(preds, true_labels)
        print("-" * 50)
        print(f"Epoch {epoch} Batch {batch_count}: Accuracy: {batch_acc:.4f}, Precision: {batch_prec:.4f}, Recall: {batch_rec:.4f}, F1: {batch_f1:.4f}")
        print(f"Loss CE: {loss_ce.item():.4f}, Score: {score.item():.4f}, Total Loss: {loss.item():.4f}")

    def train_epoch(self, epoch):
        """
        Runs one training epoch.
        """
        print()
        print()
        print()
        print()
        print()
        print()
        print()
        print()
        print()
        self.model.train()
        print(f"Total number of batches: {len(self.train_loader)}")
        batch_count = 0

        for batch in self.train_loader:
            images, labels = self.parse_batch(batch)
            self.forward_backward(images, labels, batch_count, epoch)
            batch_count += 1

            # Evaluate on validation every 50 iterations.
            if batch_count % self.eval_interval == 0:  # Changed from 50 to self.eval_interval
                print(f"\n--- Evaluation after {batch_count} iterations ---")
                val_acc = self.evaluate(self.val_loader)
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_model_state = self.model.prompt_learner.state_dict()
                    print(f"New best prompt learner found at batch {batch_count} with validation accuracy: {val_acc:.4f}")
                print("\n")

    def evaluate(self,eval_loader):
        """
        Evaluates the model on the validation set.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = len(eval_loader)
        eval_prediction, eval_labels = [], []

        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Validation", unit="batch"):
                images, labels = self.parse_batch(batch)
                logits, score = self.model(images)
                loss_ce = F.cross_entropy(logits, labels)
                loss = loss_ce + self.score_weight * score

                preds = logits.argmax(dim=1)
                eval_prediction.extend(preds.cpu().tolist())
                eval_labels.extend(labels.cpu().tolist())
                total_loss += loss.item()

        avg_loss = total_loss / num_batches
        eval_acc, eval_prec, eval_rec, eval_f1 = calculate_metrics(
            torch.tensor(eval_prediction), torch.tensor(eval_labels)
        )

        print(f"Validation Loss: {avg_loss:.4f}")
        print("Evaluation Metrics:")
        print(f"  Accuracy:  {eval_acc:.4f}")
        print(f"  Precision: {eval_prec:.4f}")
        print(f"  Recall:    {eval_rec:.4f}")
        print(f"  F1 Score:  {eval_f1:.4f}")
        return eval_acc

    def run(self):
        """
        Runs the complete training and evaluation loop.
        """
        self.print_setup()
        for epoch in range(1, self.num_epochs + 1):
            print(f"\n===== Starting Epoch {epoch} =====")
            self.train_epoch(epoch)
            val_acc = self.evaluate(self.val_loader)
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_state = self.model.prompt_learner.state_dict()
                print(f"New best prompt learner at epoch {epoch} with validation accuracy: {val_acc:.4f}")

        # Save final prompt learner and best prompt learner.
        last_model_path = os.path.join(self.output_dir, "last_prompt_learner.pth")
        best_model_path = os.path.join(self.output_dir, "best_prompt_learner.pth")
        torch.save(self.model.prompt_learner.state_dict(), last_model_path)
        if self.best_model_state is not None:
            torch.save(self.best_model_state, best_model_path)
        print(f"Prompt learner models saved to: {self.output_dir}")



# def train_with_domain_prompt(plip, plip_processor, dataloader, classnames, domain_num, device=None, val_ratio=0.1, 
#                             num_context_tokens=4, agg_vector=None, num_epochs=1, lr=5e-5, score_weight=1.0):

#     dataset = dataloader.dataset
#     total_size = len(dataset)

#     # Compute sizes for training and validation subsets.
#     val_size = int(val_ratio * total_size)
#     train_size = total_size - val_size

#     # Split the dataset into training and validation subsets.
#     train_subset, val_subset = random_split(dataset, [train_size, val_size])

#     # Create new DataLoaders with the same settings as the original.
#     train_loader = DataLoader(train_subset, batch_size=dataloader.batch_size, shuffle=True, num_workers=dataloader.num_workers)
#     val_loader = DataLoader(val_subset, batch_size=dataloader.batch_size, shuffle=False, num_workers=dataloader.num_workers)

#     # Define the computing device if not provided.
#     if device is None:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Initialize the model, passing agg_vector and num_context_tokens.
#     model = CustomCLIP(classnames=classnames, clip_model=plip, processor=plip_processor, num_context_tokens=num_context_tokens, agg_vector=agg_vector)

#     # Set up the trainer and start training.
#     trainer = PromptLearningTrainer(
#         model, train_loader, val_loader, device, domain_num,  # Add domain_num here
#         num_epochs=num_epochs, lr=lr, score_weight=score_weight
#     )
#     trainer.run()
    
    
    
def train_with_domain_prompt(plip, plip_processor, dataloader, classnames, domain_num, device=None, val_ratio=0.1, 
                             num_context_tokens=4, agg_vector=None, num_epochs=1, lr=5e-5, score_weight=1.0,
                             use_learnable_agg=False, eval_interval=50): 
    dataset = dataloader.dataset
    total_size = len(dataset)
    val_size = int(val_ratio * total_size)
    train_size = total_size - val_size

    train_subset, val_subset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_subset, batch_size=dataloader.batch_size, shuffle=True, num_workers=dataloader.num_workers)
    val_loader = DataLoader(val_subset, batch_size=dataloader.batch_size, shuffle=False, num_workers=dataloader.num_workers)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Pass the extra flag to choose which prompt learner variant to use.
    model = CustomCLIP(
        classnames=classnames, clip_model=plip, processor=plip_processor, 
        num_context_tokens=num_context_tokens, agg_vector=agg_vector, use_learnable_agg=use_learnable_agg
    )

    trainer = PromptLearningTrainer(
        model, train_loader, val_loader, device, domain_num,
        num_epochs=num_epochs, lr=lr, score_weight=score_weight,
        eval_interval=eval_interval  # Pass eval_interval here
    )
    trainer.run()

    
    
    

def train_with_domain_prompt_diff_val(plip, plip_processor, train_loader, classnames, domain_num, device=None, val_loader=None, 
                             num_context_tokens=4, agg_vector=None, num_epochs=1, lr=5e-5, score_weight=1.0,
                             use_learnable_agg=False, eval_interval=50,prefix = None): 

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Pass the extra flag to choose which prompt learner variant to use.
    model = CustomCLIP(
        classnames=classnames, clip_model=plip, processor=plip_processor, 
        num_context_tokens=num_context_tokens, agg_vector=agg_vector, use_learnable_agg=use_learnable_agg
    )

    trainer = PromptLearningTrainer(
        model, train_loader, val_loader, device, domain_num,
        num_epochs=num_epochs, lr=lr, score_weight=score_weight,
        eval_interval=eval_interval,  # Pass eval_interval here
        prefix=prefix
    )
    trainer.run()

    
    
def zero_shot_classification(loader,text_features,processor,model,device):
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




def train_with_patient_prompt(plip, plip_processor, train_loader, classnames, device=None, val_loader=None, 
                             num_context_tokens=4, agg_vector=None, num_epochs=1, lr=5e-5, score_weight=1.0,
                             use_learnable_agg=False, eval_interval=50,prefix = None,patient_id=None): 

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Pass the extra flag to choose which prompt learner variant to use.
    model = CustomCLIP(
        classnames=classnames, clip_model=plip, processor=plip_processor, 
        num_context_tokens=num_context_tokens, agg_vector=agg_vector, use_learnable_agg=use_learnable_agg
    )

    trainer = PromptLearningTrainer(
        model, train_loader, val_loader, device,
        patient_id=patient_id,  # Pass patient ID
        num_epochs=num_epochs, lr=lr, score_weight=score_weight,
        eval_interval=eval_interval,
        prefix=prefix
    )
    
    trainer.run()

    
    