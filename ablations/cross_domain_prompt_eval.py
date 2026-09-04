#!/usr/bin/env python
"""Cross-domain-prompt zero-shot generalization sweep.

For each (vlm, k) and each domain's OWN learned DIPT prompt (not aggregated
across domains), evaluate zero-shot classification on the *other* non-
validation domains, then average the per-domain metrics into a single
accuracy/precision/recall/F1 for that domain prompt.

Image features are cached once per (vlm, domain) and reused across every
(k, prompt_domain) combination, since they don't depend on which text prompt
is being tested -- this avoids re-running the image encoder ~9x more than
needed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/Amir/dipt-cpath")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from dipt.data.loaders import build_loader  # noqa: E402
from dipt.data.registry import get_dataset_spec  # noqa: E402
from dipt.evaluation.metrics import compute_metrics  # noqa: E402
from dipt.models.preprocessing import clip_preprocess  # noqa: E402
from dipt.models.vlm import load_vlm  # noqa: E402
from dipt.prompts.learned import aggregate_learned_prompts  # noqa: E402
from dipt.utils.misc import select_device  # noqa: E402

DATASET = "camelyon17"
VLMS = ["plip", "quiltnet"]
KS = [3, 4, 5]
DATA_ROOT = "/workspace/Amir/dipt-cpath/data"
PROMPT_ROOT = "/workspace/Amir/dipt-cpath/outputs/prompts"
BATCH_SIZE = 256
NUM_WORKERS = 8
GPU = 0
OUT_JSON = "/workspace/Amir/dipt-cpath/ablations/cross_domain_prompt_results.json"


def main() -> None:
    spec = get_dataset_spec(DATASET)
    device = select_device(GPU)
    domains = [d for d in spec.domains if d != spec.val_domain]
    print(f"non-validation domains: {domains}", flush=True)

    results: dict[str, dict] = {}

    for vlm in VLMS:
        print(f"\n===== loading {vlm} =====", flush=True)
        model, processor, vlm_spec = load_vlm(vlm, device=device, freeze=True)
        model.eval()

        cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for domain in domains:
            loader = build_loader(
                spec, [domain], BATCH_SIZE, shuffle=False,
                data_root=DATA_ROOT, num_workers=NUM_WORKERS,
            )
            feats, labels = [], []
            with torch.no_grad():
                for batch in loader:
                    images, y = batch[0].to(device), batch[1]
                    pixel_values = clip_preprocess(images)
                    f = F.normalize(model.get_image_features(pixel_values=pixel_values), dim=-1)
                    feats.append(f)
                    labels.append(y)
            cache[domain] = (torch.cat(feats), torch.cat(labels).to(device))
            print(f"[{vlm}] cached domain {domain}: {cache[domain][0].shape[0]:,} images", flush=True)

        for k in KS:
            for prompt_domain in domains:
                try:
                    text_features = aggregate_learned_prompts(
                        spec.classnames, model, processor, k, [prompt_domain],
                        DATASET, vlm, prompt_root=PROMPT_ROOT, device=device,
                    )
                except FileNotFoundError as exc:
                    print(f"SKIP {vlm} k={k} domain{prompt_domain}: {exc}", flush=True)
                    continue

                eval_domains = [d for d in domains if d != prompt_domain]
                per_domain = {}
                for d in eval_domains:
                    image_features, labels = cache[d]
                    logits = (100.0 * image_features @ text_features.T).float()
                    preds = logits.argmax(dim=1).cpu()
                    per_domain[d] = compute_metrics(preds, labels.cpu()).as_dict()

                avg = {
                    key: sum(m[key] for m in per_domain.values()) / len(per_domain)
                    for key in ("accuracy", "precision", "recall", "f1")
                }

                key = f"{vlm}_k{k}_domain{prompt_domain}"
                results[key] = {
                    "vlm": vlm,
                    "k": k,
                    "prompt_domain": prompt_domain,
                    "eval_domains": eval_domains,
                    "per_domain": per_domain,
                    "average": avg,
                }
                print(
                    f"{key}: eval_on={eval_domains} "
                    f"avg_acc={avg['accuracy']*100:.2f} avg_prec={avg['precision']*100:.2f} "
                    f"avg_rec={avg['recall']*100:.2f} avg_f1={avg['f1']*100:.2f}",
                    flush=True,
                )

        del model, processor, cache
        torch.cuda.empty_cache()

    with open(OUT_JSON, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"\nSaved {len(results)} rows to {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
