# DIPT: Domain Invariant Prompt Tuning for Knowledge Distillation in Computational Pathology

## Overview

Computational Pathology (CPath) models often struggle with domain shifts caused by variations in staining protocols, scanner devices, and imaging settings. Our project addresses this challenge by leveraging vision-language models (VLMs)—specifically PLIP (a pathology-tuned CLIP)—as a robust source for knowledge distillation. Predefined prompts for zero-shot inference are insufficient due to their sensitivity to prompt variations and the lack of semantic descriptors for histopathological data.

## Key Contributions

- **Domain Invariant Prompt Tuning (DIPT):**  
  We propose DIPT, a novel prompt tuning step that learns domain-specific tokens via prefix tuning and then aggregates them to form domain-invariant, class-generic prompts. These prompts guide the knowledge distillation process, enabling the student model to capture robust and generalizable features.

- **Improved Domain Generalization:**  
  By aligning the student’s visual features with domain-invariant embeddings from PLIP’s text encoder, our approach achieves improved performance. Our experiments demonstrate up to a 7.7% F1-score improvement on challenging datasets such as CAMELYON17-WILDS and Kather19.

## Method Summary

1. **Prompt Tuning:**  
   Learnable tokens are concatenated with a fixed generic token and tuned via prefix tuning to capture domain-specific variations. These tokens are then averaged across domains to generate domain-invariant, class-generic embeddings.

2. **Knowledge Distillation:**  
   The domain-invariant prompts are used to distill knowledge from both the PLIP image and text encoders. This dual-modality distillation helps the student model learn a robust representation suitable for various unseen pathology domains.

