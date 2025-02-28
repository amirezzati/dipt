import torchvision.transforms as transforms
import numpy as np

from collections import defaultdict
from pathlib import Path
import csv
import torch
import os
import sys


from transformers import CLIPProcessor, CLIPModel
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import timm
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*torch.load with weights_only=False.*")

from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.query import Q     ## query
from domainbed.lib.logger import Logger


sys.path.append(os.path.abspath('../'))
from datasets import MultipleDomainDataset

import PlipTrain

sys.path.append(os.path.abspath('./'))
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable tokenizers parallelism


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

test_dataset = MultipleDomainDataset(root_dir=root_data_dir, domains=['4'],  # Specify domains to include
    transform=transform
)

batch_size = 128
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)




print(next(iter(test_loader))[0].shape)
sample_image = test_dataset[0][0]


dims = {
    "RN50": 512, # former: 1024
    "RN101": 512,
    "ViT-B/32": 512,
    "ViT-B/16": 512,
    "ViT-L/14": 768,
}

args['name'] = "PLIP-VL2V-ADiP_1"
args['gpu_id'] = 0
args['dataset_name'] = "camelyon17"
args['lambda'] = 0.5
args['prompt_sytle'] = 2
args['backbone'] = "ViT-B/16"    # "ViT-L/14" "vit-B/16"
args['model'] = "vit-base"
args['pretrained'] = True  # student model must preserve its pretrained weights

args['pretrained_path'] =  Path("./train_output") / args['dataset_name'] / args['name'] / 'models' 
# args["model"] = used for resnet

lr_list = np.array([5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3])

# # args["optimizer"] = 'adamw'
# args["optimizer"] = ("adam", "adam")
# random_state = np.random.RandomState(0)
# # args["lr"] = (5e-5, random_state.choice(lr_list))
# args["lr"] = 5e-5
# # args["weight_decay"] = (0.0, 10 ** random_state.uniform(-6, -2))
# args["weight_decay"] = 10 ** random_state.uniform(-6, -2)
# args['nsteps'] =  3000

args["work_dir"] = '.'
output_dir = args["work_dir"] / Path("train_output") 
output_dir.mkdir(exist_ok=True, parents=True)

args["out_root"] = args["work_dir"] / Path("train_output") / args["dataset_name"]
args["out_root"].mkdir(exist_ok=True, parents=True)

args["out_dir"] = args["out_root"] / args["name"]
args["out_dir"].mkdir(exist_ok=True, parents=True)

args["model_save"] = True
args["save_step"] = 200



# initialize data dict
data = {}
data['name'] = "camelyon17"
data['num_classes'] = 2
data['num_domains'] = 5
data['class_names'] = ['healthy', 'canserous']
print(sample_image.shape)
data['input_shape'] = sample_image.shape
data['train_doms'] = ["0", "1", "2"]
data['val_doms'] = ["3"]
data['test_doms'] = ["4"]



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
train_doms = [["0", "1", "2"], ["0", "1", "3"]]
val_doms = ["3", "2"]
test_dom = ["4"]

transform = transforms.Compose(
    [
    transforms.Resize(224),
    transforms.ToTensor(),
    ]
)

logger = Logger.get(args["out_dir"] / "log.txt")


plip_model = algorithms.PLIP(args)


batch_size = 128

def inference_all_plip(plip_model, args, data, train_doms, val_doms, transform, batch_size, logger=None):
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

        stage = 3

        print(f"Stage {stage}: Training on domains {train_domains}, Validating on domain {val_domain}")

        root_data_dir = args['data_dir'] + '/camelyon17d'

        print(root_dir)
        test_dataset = MultipleDomainDataset(root_dir=root_data_dir, domains=['4'], transform=transform)

        # Create DataLoader
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

        res, records = PlipTrain.inference(stage, test_loader=test_loader, teacher=plip_model, args=args, data=data, logger=logger)
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
    print('aggregated resluts:')
    for key in mean_results:
        logger.info(key + " is: " + str(mean_results[key]))



    # csv_file = "mean_results.csv"
    # with open(csv_file, mode='w', newline='') as file:
    #     writer = csv.DictWriter(file, fieldnames=mean_results.keys())
        
    #     # Write the header
    #     writer.writeheader()
        
    #     # Write the mean values
    #     writer.writerow(mean_results)

    # print(f"Mean results saved to {csv_file}")

    filename = "aggregated_test_records.csv"
    save_path = Path(args["out_root"]) / args["name"] / 'records'
    save_records(mean_results, save_path, filename=filename)


    return mean_results

inference_all_plip(plip_model, args, data, train_doms, val_doms, transform, batch_size, logger=logger)
