# from wilds import get_dataset
# from wilds.common.data_loaders import get_train_loader



import numpy as np
from itertools import chain

import collections
import time


import torch
# import pandas as pd
from tqdm import tqdm
# from sconf import Config
from torch import nn
import collections


import os
import sys
import copy

from transformers import CLIPProcessor, CLIPModel
from torch.nn import functional as F


sys.path.append(os.path.abspath('../'))
import Prompts.load_prompts as pt ## loading prompts
# import PlipTrain


# from domainbed import algorithms
from domainbed.optimizers import get_optimizer
from domainbed.networks.backbones import get_backbone # get backbone used for knowledge distillation.
from domainbed.lib import misc
from domainbed.lib.query import Q     ## query


from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

sys.path.append(os.path.abspath('../'))
from datasets import MultipleDomainDataset, CyclicDataLoader

import warnings
# warnings.filterwarnings("ignore", category=UserWarning, message=".*torch.load with weights_only=False.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.load.*weights_only=False.*")


dims = {
    "RN50": 512,  # former: 1024
    "RN101": 512,
    "ViT-B/32": 512,
    "ViT-B/16": 512,
    "ViT-L/14": 768,
}


# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved



class ClassificationHead(torch.nn.Linear):
    def __init__(self, normalize, weights, biases=None):
        output_size, input_size = weights.shape
        super().__init__(input_size, output_size)
        self.normalize = normalize
        if weights is not None:
            self.weight = torch.nn.Parameter(weights.clone())
        if biases is not None:
            self.bias = torch.nn.Parameter(biases.clone())
        else:
            self.bias = torch.nn.Parameter(torch.zeros_like(self.bias))

    def forward(self, inputs):
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return super().forward(inputs)



class PLIP:
    """PLIP interface to save memory during algorithm save"""

    def __init__(self, hparams):
        # Use GPU if available, otherwise CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Device is: ", self.device)
        self.hparams = hparams

        # Load the PLIP model and processor
        self.plip_model = CLIPModel.from_pretrained("vinid/plip")
        self.processor = CLIPProcessor.from_pretrained("vinid/plip")
        
        self.plip_model = self.plip_model.to(self.device)
        self.plip_model.eval()
        for param in self.plip_model.parameters():
            param.requires_grad = False

    def get_img_feat(self, x):
        """Get normalized image embeddings"""

        with torch.no_grad():
            clip_img_feat = self.plip_model.get_image_features(x)
            clip_img_feat /= clip_img_feat.norm(dim=-1, keepdim=True)
        return clip_img_feat

    def get_txt_feat(self, labels):
        """Get normalized text embeddings"""

        with torch.no_grad():
            clip_txt_feat = []
            for i in range(labels.size(0)):
                feat = self.zeroshot_weights[labels[i].item()]
                clip_txt_feat.append(feat)
            clip_txt_feat = torch.stack(clip_txt_feat, dim=0)
        return clip_txt_feat

    def get_zeroshot_classifier(self, classnames, templates):

        # logit_scale = self.plip_model.config.logit_scale
        logit_scale = self.plip_model.logit_scale

        print("Getting zeroshot weights.")
        with torch.no_grad():
            zeroshot_weights = []
            for classname in tqdm(classnames):
                texts = [
                    template.format(class_name=classname) for template in templates
                ]

                # Process text inputs for embeddings
                inputs = self.processor(
                    text=texts, images=None, return_tensors="pt", padding=True
                ).to(self.device)

                embeddings = self.plip_model.get_text_features(**inputs)  # Embed with text encoder
                embeddings /= embeddings.norm(dim=-1, keepdim=True)
                embeddings = embeddings.mean(dim=0, keepdim=True)
                embeddings /= embeddings.norm()
                zeroshot_weights.append(embeddings)

            # Compute zero-shot weights
            zeroshot_weights = torch.stack(zeroshot_weights, dim=0).to(self.device)
            zeroshot_weights = torch.transpose(zeroshot_weights, 0, 2)
            zeroshot_weights *= logit_scale.exp()
            zeroshot_weights = zeroshot_weights.squeeze().float()
            zeroshot_weights = torch.transpose(zeroshot_weights, 0, 1)

        classification_head = ClassificationHead(
            normalize=True, weights=zeroshot_weights
        )
        self.zeroshot_weights = zeroshot_weights
        return classification_head, zeroshot_weights




class PLIPP:
    """PLIP interface to save memory during algorithm save"""

    def __init__(self, hparams, prompts_embedding = None):
        # Use GPU if available, otherwise CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Device is: ", self.device)
        self.hparams = hparams

        # Load the PLIP model and processor
        self.plip_model = CLIPModel.from_pretrained("vinid/plip")
        self.processor = CLIPProcessor.from_pretrained("vinid/plip")
        
        self.plip_model = self.plip_model.to(self.device)
        self.plip_model.eval()
        for param in self.plip_model.parameters():
            param.requires_grad = False

        self.learned_prompt_embeddings = prompts_embedding
        

    def get_img_feat(self, x):
        """Get normalized image embeddings"""

        with torch.no_grad():
            clip_img_feat = self.plip_model.get_image_features(x)
            clip_img_feat /= clip_img_feat.norm(dim=-1, keepdim=True)
        return clip_img_feat

    def get_txt_feat(self, labels):
        """Get normalized text embeddings"""

        with torch.no_grad():
            clip_txt_feat = []
            for i in range(labels.size(0)):
                feat = self.zeroshot_weights[labels[i].item()]
                clip_txt_feat.append(feat)
            clip_txt_feat = torch.stack(clip_txt_feat, dim=0)
        return clip_txt_feat

    def get_zeroshot_classifier(self, classnames, templates):

        # logit_scale = self.plip_model.config.logit_scale
        logit_scale = self.plip_model.logit_scale

        print("Getting zeroshot weights.")
        with torch.no_grad():
            zeroshot_weights = self.learned_prompt_embeddings
            zeroshot_weights *= logit_scale.exp()

        classification_head = ClassificationHead(
            normalize=True, weights=zeroshot_weights
        )
        self.zeroshot_weights = zeroshot_weights
        return classification_head, zeroshot_weights


class BACKBONE(torch.nn.Module):
    """backbone for student models"""

    def __init__(self, input_shape, hparams):
        super(BACKBONE, self).__init__()
        if hparams["pretrained"] == "False":
            self.network, self.n_outputs = get_backbone(
                hparams["model"], preserve_readout=False, pretrained=False
            )
        else:
            self.network, self.n_outputs = get_backbone(
                hparams["model"], preserve_readout=False, pretrained=True
            )
        self.hparams = hparams

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        output = self.network(x)
        return output

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        # self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()



def Featurizer(input_shape, hparams, **kwargs):
    """Auto-select an appropriate featurizer for the given input shape."""

    if input_shape[1:3] == (224, 224):
        # if hparams["backbone"].startswith("ViT") or hparams["backbone"].startswith("deit"):
        return BACKBONE(input_shape, hparams)
        # return ResNet(input_shape, hparams, **kwargs)
    else:
        raise NotImplementedError(f"Input shape {input_shape} is not supported")



class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """

    transforms = {}

    def __init__(self, input_shape, args, data): # remove input_shape or not?????
        # super(Algorithm, self).__init__()
        super().__init__()
        self.input_shape = input_shape
        # self.num_classes = num_classes
        # self.num_domains = num_domains
        self.input_shape = data['input_shape']

        self.num_classes = data['num_classes']
        self.num_domains = data['num_domains']
        self.args = args


    # def update(self, x, y, **kwargs):
    def update(self, x, y, teacher_model):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

    def forward(self, x):
        return self.predict(x)

    def new_optimizer(self, parameters):
        optimizer = get_optimizer(
            self.args["optimizer"],
            parameters,
            lr=self.args["lr"],
            weight_decay=self.args["weight_decay"],
        )
        # print('used Algorithm class to make a new optimizer')
        return optimizer

    def clone(self):
        clone = copy.deepcopy(self)
        clone.optimizer = self.new_optimizer(clone.network.parameters())
        clone.optimizer.load_state_dict(self.optimizer.state_dict())

        return clone
    


    # todo: generate prompts:

class KnowledgeDistillation(Algorithm):
    """Distillation from CLIP"""

    def __init__(
        self, teacher, input_shape, args, data
    ):
        super().__init__(
            input_shape, args, data
        )

        ##########################
        # CHANGES TO MAKE FOR ID
        ##########################
        """
        1. SET FEATURIZER = timm.models.vit_base_patch_16_224
        2. IGNORE num_domains, hparams, tgt_dom
        3. set lmd=0.1
        4. prompt_style = 2
        5. get classnames from your ID code
        """
        ########################## target_dom deleted from prototype

        self.args = args
        self.lmd = args["lambda"]
        # self.cls_num = data -> no need -> in the father
        # self.dom_num = num_domains + 1 # why???????????????
        # self.num_domains = args['num'] -> no need -> in the father

        self.prompt_style = args['prompt_sytle']
        
        # type 1 - {class}
        if self.prompt_style == 1:
            print("Prompt Template: \{class\}")
            self.templates = ["{class_name}"]

        # type 2 - a photo of a {class}
        elif self.prompt_style == 2:
            print("Prompt Template: a photo of a \{class\}")
            self.templates = ["a photo of a {class_name}"]

        # type 3 - a {dom} photo of a {class}
        elif self.prompt_style == 3:
            print("Prompt Template: a \{dom\} photo of a \{class\}")
            self.templates = [
                "an art of a {class_name}",
                "a clipart of a {class_name}",
                "a photo of a {class_name} product",
                "a photo of a {class_name}",
            ]

        # type 4 - a {dom} photo of a {small / big} {class}
        elif self.prompt_style == 4:
            print("Prompt Template: a \{dom\} photo of a \{small/big\} \{class\}")
            self.templates = [
                "an art photo of a {size} {{class_name}}",
                "a clipart photo of a {size} {{class_name}}",
                "a product photo of a {size} {{class_name}}",
                "a real photo of a {size} {{class_name}}",
            ]

        # type 5 - a {dom} photo of a {class} w/o target
        # elif self.prompt_style == 5:
        #     print("Prompt Template: a \{dom\} photo of a \{class\} w/o target")
        #     self.templates = [
        #         "an art of a {class_name}",
        #         "a clipart of a {class_name}",
        #         "a photo of a {class_name} product",
        #         "a photo of a {class_name}",
        #     ]
        #     del self.templates[data[target_dom]]

        # type 6 - a photo of a {class_label}
        
        elif self.prompt_style == 6:
            # classnames = [str(i) for i in range(len(classnames))]
            print("Prompt Template: a photo of a \{class_label\}")
            self.templates = ["a photo of a {class_name}"]



        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.student = Featurizer(input_shape, self.args)

        # [#] zero-shot classifier from CLIP
        # print('cls names: ', args["class_names"])
        # Loading CLIP model

        classnames = [name.replace("_", " ") for name in data["class_names"]]
        print('cls names: ', data["class_names"])

        self.clip_cls, self.zeroshot_weights = teacher.get_zeroshot_classifier(
            classnames, self.templates
        )



class VL2V_ADiP(KnowledgeDistillation):

    @staticmethod
    def dfc_loss(img, txt, proj, lmd):
        kd_loss_1 = -torch.mean(F.cosine_similarity(proj, img))
        kd_loss_2 = -torch.mean(F.cosine_similarity(proj, txt))
        kd_loss = lmd * kd_loss_1 + (1 - lmd) * kd_loss_2
        return kd_loss

    @staticmethod
    def rand_bbox(size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def __init__(self, stage, teacher, input_shape, args, data):
        super(VL2V_ADiP, self).__init__(
            teacher, input_shape, args, data
        )

        """
        1. set embed_dim=512
        """
        
        self.stage = stage
        self.embed_dim = dims[args["backbone"]]

        print('==========================self.embed_dim:', self.embed_dim)
        print('==========================self.student.n_outputs:', self.student.n_outputs)
        self.classifier = self.clip_cls
        self.proj_lyr = nn.Linear(self.student.n_outputs, self.embed_dim)
        # choose between stages
        if self.stage == 1:
            train_params = chain(
                self.proj_lyr.parameters(),
            )
        elif self.stage ==2:
            train_params = chain(
            self.student.parameters(),
        )
        if stage != 3:  # stage 3 is inference
            self.optimizer = torch.optim.Adam(
                train_params,
                lr=self.args["lr"],
                weight_decay=self.args["weight_decay"],
            )
        self.stage = stage

    # def update(self, x, y, **kwargs):
    def update(self, x, y, teacher_model):
        # all_x = torch.cat(x)
        # all_y = torch.cat(y)
        all_x = x
        all_y = y

        # # [#] Forward pass

        r = np.random.rand(1)

        # [#] normal forward pass
        # clip features

        # clip_img_feat = kwargs["clip_model"].get_img_feat(all_x)
        # clip_txt_feat = kwargs["clip_model"].get_txt_feat(all_y)
        clip_img_feat = teacher_model.get_img_feat(all_x)
        clip_txt_feat = teacher_model.get_txt_feat(all_y)
        

        # forward pass
        img_feat = self.student(all_x)
        proj_feat = self.proj_lyr(img_feat)
        proj_feat_norm = proj_feat / proj_feat.norm(dim=-1, keepdim=True)

        # loss comp.
        loss = self.dfc_loss(clip_img_feat, clip_txt_feat, proj_feat_norm, self.lmd)

        # [#] Backward pass

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}

    def predict(self, x):
        x = self.student(x)
        x = self.proj_lyr(x)
        return self.classifier(x)

    def forward(self, x):
        return self.predict(x)



def load_algorithm(stage, algorithm, args, data, logger, last = False):
    train_dom_names = ''.join(map(str, data["train_doms"]))

    filename = "t_" + train_dom_names + "v_" + data["val_doms"] + f'_stage{stage-1}' + "_best.pth"
    last_filename = "last_t_" + train_dom_names + "v_" + data["val_doms"] + f'_stage{stage-1}' + "_best.pth"

    model_path = args["pretrained_path"] / filename
    last_model_path = args["pretrained_path"] / last_filename

    if stage == 1:
        logger.info("===== stage 1 =====")


    elif stage == 2: 
        logger.info("===== stage 2 =====")

        if os.path.exists(model_path):
            print(f"loading model for stage{stage} and path: {model_path}")
            model_dict = torch.load(model_path, weights_only=False)
        else:
            print(f"Model not found at {model_path}. Loading from fallback path: {last_model_path}")
            model_dict = torch.load(last_model_path, weights_only=False)


        algorithm_dict = model_dict["model_dict"]
        algorithm.load_state_dict(algorithm_dict, strict=True)

    else: 
        
        logger.info("===== stage 3 =====")
        # if last: 
        #     print(f"last model reloading from fallback path: {last_model_path}")
        #     model_dict = torch.load(last_model_path)
        # elif os.path.exists(model_path):
        #     print(f"loading model for stage{stage} and path: {model_path}")
        #     model_dict = torch.load(model_path)
        # else:
        #     print(f"Model not found at {model_path}. Loading from fallback path: {last_model_path}")
        #     model_dict = torch.load(last_model_path)

        if not last:
            print(f"=== best === loading model for stage{stage} and path: {model_path}")
            model_dict = torch.load(model_path, weights_only=False)
        else:
            print(f"=== last === Loading from fallback path: {last_model_path}")
            model_dict = torch.load(last_model_path, weights_only=False)

        
        model_dict = torch.load(model_path, weights_only=False)
        algorithm_dict = model_dict["model_dict"]
        algorithm.load_state_dict(algorithm_dict, strict=True)


    return algorithm




# from domainbed.lib.fast_data_loader import FastDataLoader

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

class Evaluator:
    def __init__(
        self,
        train_loader,
        val_loader,
        args,
        data,
        logger,
        debug=False,
        test = False,
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        
        # all_envs = list(range(n_envs))
        if test:
            self.train_envs = []
            self.test_envs = data['test_doms']
            self.num_iter =  len(val_loader)
        else: 
            self.train_envs = data['train_doms']
            self.test_envs = data['val_doms']
            self.num_iter = 10

        self.classnames = data['class_names']
        # self.n_envs = n_envs
        self.logger = logger
        self.debug = debug
        self.test = test



    def calc_metrics(self, algorithm, iterator, test, debug=False):
        correct = 0
        total = 0
        losssum = 0.0

        # Initialize lists to store predictions and true labels
        all_preds = []
        all_labels = []

        algorithm.eval()

        # print(f'in calc metric function:  len of iter or number of batches {len(iterator)}')
        # print(f'and len num of iteration is:', self.num_iter)
        # i = 0
        # for i in range(self.num_iter):
        for i, batch in tqdm(enumerate(iterator), desc="EVAL",  total=len(iterator)):

            x = batch[0].to(device)
            y = batch[1].to(device)

            with torch.no_grad():
                logits = algorithm.predict(x)
                loss = F.cross_entropy(logits, y).item()

            B = len(x)
            losssum += loss

            # Get predictions
            preds = logits.argmax(1).cpu().numpy()
            labels = y.cpu().numpy()

            # Store predictions and true labels
            all_preds.extend(preds)
            all_labels.extend(labels)

            correct += (preds == labels).sum()
            total += B

            # if self.test and i % 10 == 0: 
            #     self.logger.info(f"accuracy in {i}/{self.num_iter}: {correct/total:.3f}")
            #     self.logger.info(f"loss in {i}/{self.num_iter}: {losssum/total:.3f}")

            if debug:
                break

        algorithm.train()

        # Calculate accuracy and loss
        acc = correct / (total + 1e-6)
        loss = losssum / (total + 13-6)

        # Calculate precision, recall, and F1 score
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        # print(confusion_matrix(all_labels, all_preds))
        
        # print("acc:================================================1", acc)
        # print("recall:================================================1", recall)
        
        return acc, precision, recall, f1, loss



    def evaluate(self, algorithm, ret_losses=False):
        # n_train_envs = len(self.train_envs)
        n_test_envs = len(self.test_envs)

        assert n_test_envs == 1

        summaries = collections.defaultdict(float)
        # for key order
        summaries["train"] = 0.0
        summaries["val"] = 0.0
        summaries["test"] = 0.0

        metrics = {}
        losses = {}

        # train_iter = iter(self.train_loader)
        val_iter = iter(self.val_loader)
        # for name, loader_kwargs, weights in self.eval_meta:
            # env\d_[in|out]
        if not self.test: 
            # name = f"train_dom{str(self.train_envs)}"
            # acc, precision, recall, f1, loss= self.calc_metrics(algorithm, iterator=self.train_iter, test = self.test)
            # metrics['train_acc'] = acc
            # metrics['train_precision'] = precision
            # metrics['train_recall'] = recall
            # metrics['train_f1'] = f1
            # losses['train_loss'] = loss
            acc, precision, recall, f1, loss = self.calc_metrics(algorithm, iterator=val_iter, test= self.test)
            metrics['val_acc'] = acc
            metrics['val_precision'] = precision
            metrics['val_recall'] = recall
            metrics['val_f1'] = f1
            losses['val_loss'] = loss

        else: 
            acc, precision, recall, f1, loss = self.calc_metrics(algorithm, iterator=val_iter, test = self.test)
            metrics["test_acc"] = acc
            metrics['test_precision'] = precision
            metrics['test_recall'] = recall
            metrics['test_f1'] = f1
            losses["test_loss"] = loss
            # print("acc:================================================", acc)
            # print("recall:================================================", recall)

        if ret_losses:
            return metrics, summaries, losses
        else:
            return metrics, summaries




def train(stage, train_loader, val_loader, args, data, checkpoint_freq, logger = None, mean_doms = False):

    assert stage == 1 or stage == 2 or stage == 3

    logger.info(f"==================== stage {stage} =======================")
    
    device = args['device']

    if args["Prompt"]:
        num_tokens = 4
        logger.info("========== loading prompts for knowledge distillation with prompts =========")
        # def get_learned_prompts(classnames,plip, plip_processor, num_context_tokens=4, doms=None, device="cuda"):
        class_names = data["class_names"]
        train_doms = data['train_doms']
        if args["val1"]:
            print('----------------------we are using dom1 as validation domain ------------------------')
            learned_prompts = pt.load_learned_prompts_val1(class_names, num_tokens, doms=train_doms, device = device)
        elif args["kather"]: 
            print('----------------------load prompts learned for kather dataset------------------------')
            learned_prompts = pt.load_learned_prompts_kather(class_names, num_tokens, doms=train_doms, device = device)
        else: 
            print('---------------------- the default prompt loader is used -------------------------')
            learned_prompts = pt.load_learned_prompts(class_names, num_tokens, doms=train_doms, device = device)

        teacher = PLIPP(args, learned_prompts)
    else:
        logger.info("========== knowledge distillation without prompts =========")
        teacher = PLIP(args)


    algorithm = VL2V_ADiP(stage, teacher=teacher, input_shape=data["input_shape"], args=args, data=data)
    algorithm.cuda()
    # def __init__(self, stage, teacher, input_shape, args, data):


    n_params = sum([p.numel() for p in algorithm.parameters()])
    logger.info("# of params = %d" % n_params)

    # train_minibatches_iterator = zip(*train_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])

    algorithm = load_algorithm(stage, algorithm, args, data, logger)


    evaluator = Evaluator(
        train_loader = train_loader,
        val_loader = val_loader,
        args = args,
        data = data,
        logger = logger,
        test = False,
    )
    
    last_results_keys = None
    records = []
    epochs_path = args["out_dir"] / "results.jsonl"


    steps_per_epoch = len(train_loader)
    

    ######## training loop #######

    n_steps = args['nsteps']
    # train_iter = iter(train_loader)
    # train_iter = cycle(iter(train_loader))
    train_iter = CyclicDataLoader(train_loader)
    
    for step in tqdm(range(n_steps), desc=f"Training"):
        # print('number of epochs:', n_steps)
        step_start_time = time.time()
        # batches_dictlist: [{env0_data_key: tensor, env0_...}, env1_..., ...]

        batch = next(train_iter)


        # print(type(batch))
        x = batch[0].cuda()
        y = batch[1].cuda()
        z = batch[2].cuda()
 
        step_vals = algorithm.update(x, y, teacher_model=teacher)
        train_dom_names = ''.join(map(str, data["train_doms"]))
        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)

        checkpoint_vals["step_time"].append(time.time() - step_start_time)

        if step % checkpoint_freq == 0 and step > 0:
            results = {
                "step": step,
                "epoch": step / steps_per_epoch,
            }

            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)

            eval_start_time = time.time()
            metrics, summaries = evaluator.evaluate(algorithm)
            results["eval_time"] = time.time() - eval_start_time

            # results = (epochs, loss, step, step_time)
            results_keys = (
                # list(summaries.keys())
                list(metrics.keys())
                + list(results.keys())
            )
            # merge results
            # results.update(summaries)
            results.update(metrics)

            # print
            if results_keys != last_results_keys:
                logger.info(misc.to_row(results_keys))
                last_results_keys = results_keys

            logger.info(misc.to_row([results[key] for key in results_keys]))
            records.append(copy.deepcopy(results))

            # [#] Model saving
            temp_rec = Q(records)
            # print(temp_rec)
            if (
                args["model_save"]
                and temp_rec.argmax("val_acc")["val_acc"]
                == temp_rec[-1]["val_acc"]
                and step >= args["save_step"]
            ):

                ckpt_dir = args["out_dir"] / "models"
                ckpt_dir.mkdir(exist_ok=True)

                # ckpt_dir = args["out_dir"] / "checkpoints" / train_dom_names
                # ckpt_dir.mkdir(exist_ok=True)


                # ############################### save file name ###############################################
                print('saving the model...')
                filename = "t_" + train_dom_names + f"v_" + data["val_doms"] + f'_stage{stage}' + "_best.pth"
                
                path = ckpt_dir / filename

                save_dict = {
                    "args": dict(args),
                    "train_env": data["train_doms"],
                    "val_env" : data["val_doms"],
                    "test_envs": data["test_doms"],
                    "model_dict": algorithm.cpu().state_dict(),
                }
                algorithm.cuda()
                torch.save(save_dict, path)

    # last save of the model: 
    filename = "last_t_" + train_dom_names + f"v_" + data["val_doms"] + f'_stage{stage}' + "_best.pth"
    
    ckpt_dir = args["out_dir"] / "models" / str()
    ckpt_dir.mkdir(exist_ok=True)

    path = ckpt_dir / filename

    save_dict = {
        "args": dict(args),
        "train_env": data["train_doms"],
        "val_env" : data["val_doms"],
        "test_envs": data["test_doms"],
        "model_dict": algorithm.cpu().state_dict(),
    }
    algorithm.cuda()
    torch.save(save_dict, path)


    # find best
    logger.info("---")

    records = Q(records)
    best_val_acc = records.argmax("val_acc")["val_acc"]
    best_val_prec = records.argmax("val_acc")["val_precision"]
    best_val_rec = records.argmax("val_acc")["val_recall"]
    best_val_f1 = records.argmax("val_acc")["val_f1"]
    last_val_acc = records[-1]["val_acc"]
    last_val_prec = records[-1]["val_precision"]
    last_val_rec = records[-1]["val_recall"]
    last_val_f1 = records[-1]["val_f1"]

    
    ret = {
    "best_acc" : best_val_acc,
    "best_val_prec" : best_val_prec ,
    "best_val_rec" : best_val_rec,
    "best_val_f1" : best_val_f1,
    "last_acc" : last_val_acc,
    "last_val_prec" : last_val_prec,
    "last_val_rec" : last_val_rec,  
    "last_val_f1" : last_val_f1,
    }

    for k, acc in ret.items():
        logger.info(f"{k} = {acc:.3%}")

    return ret, records




def inference(stage, test_loader, args, data, logger = None, last = False):

    assert stage == 3  # stage 3 is for inference
    
    device = args['device']

    if args["Prompt"]:
        num_tokens = 4
        class_names = data["class_names"]
        train_doms = data['train_doms']
        if args["val1"]:
            print('----------------------we are using dom1 as validation domain ------------------------')
            learned_prompts = pt.load_learned_prompts_val1(class_names, num_tokens, doms=train_doms, device=device)
        elif args["kather"]: 
            print('----------------------load prompts learned for kather dataset------------------------')
            learned_prompts = pt.load_learned_prompts_kather(class_names, num_tokens, doms=train_doms, device = device)
        else: 
            print('---------------------- the default prompt loader is used -------------------------')
            learned_prompts = pt.load_learned_prompts(class_names, num_tokens, doms=train_doms, device = device)
            
        teacher = PLIPP(args, learned_prompts)
    else:
        teacher = PLIP(args)

    algorithm = VL2V_ADiP(stage, teacher=teacher, input_shape=data["input_shape"], args=args, data=data)
    algorithm.cuda()

    n_params = sum([p.numel() for p in algorithm.parameters()])
    logger.info("# of params = %d" % n_params)

    checkpoint_vals = collections.defaultdict(lambda: [])

    algorithm = load_algorithm(stage, algorithm, args, data, logger, last)


    evaluator = Evaluator(
        train_loader = None,
        val_loader = test_loader,
        args = args,
        data = data,
        logger = logger,
        test = True,
    )
    
    last_results_keys = None

    records = []    
    results = {}

    for key, val in checkpoint_vals.items():
        results[key] = np.mean(val)

    eval_start_time = time.time()
    metrics, summaries = evaluator.evaluate(algorithm)
    results["eval_time"] = time.time() - eval_start_time

    results_keys = (
        # list(summaries.keys())
        list(metrics.keys())
        + list(results.keys())
    )

    results.update(metrics)

    # print
    if results_keys != last_results_keys:
        logger.info(misc.to_row(results_keys))
        last_results_keys = results_keys

    logger.info(misc.to_row([results[key] for key in results_keys]))
    records.append(copy.deepcopy(results))

    # find best
    logger.info("---")

    records = Q(records)
    best_val_acc = records[0]["test_acc"]
    best_val_prec = records[0]["test_precision"]
    best_val_rec = records[0]["test_recall"]
    best_val_f1 = records[0]["test_f1"]
    # last_val_acc = records[-1]["val_acc"]
    # last_val_prec = records[-1]["val_precision"]
    # last_val_rec = records[-1]["val_recall"]
    # last_val_f1 = records[-1]["val_f1"]

    
    ret = {
    "acc" : best_val_acc,
    "test_prec" : best_val_prec ,
    "test_rec" : best_val_rec,
    "test_f1" : best_val_f1,
    # "last_acc" : last_val_acc,
    # "last_val_prec" : last_val_prec,
    # "last_val_rec" : last_val_rec,  
    # "last_val_f1" : last_val_f1,
    }

    for k, acc in ret.items():
        logger.info(f"{k} = {acc:.3%}")

    return ret, records