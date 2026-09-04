"""DIPT prompt learner.

A prompt for class *i* is the token sequence::

    [SOT]  c_1 ... c_K  a_i  [EOT] ...

where ``c_1..c_K`` are K learnable *domain-specific* context tokens and ``a_i`` is
a frozen *class-generic* token: the aggregated embedding of hand-written
templates for class *i* (see :mod:`dipt.prompts.templates`).

Only the context tokens are optimised. The resulting per-domain prompts are then
averaged in embedding space to obtain the domain-invariant prompts consumed by
the knowledge-distillation stage.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor

from dipt.models.preprocessing import clip_preprocess


class TextEncoder(nn.Module):
    """CLIP text transformer applied to *embedded* tokens instead of token ids.

    Running the encoder on embeddings is what lets gradients reach the learnable
    context vectors.
    """

    def __init__(self, clip_model: CLIPModel) -> None:
        super().__init__()
        self.clip_model = clip_model
        self.dtype = clip_model.dtype

    def forward(self, embedded_tokens: torch.Tensor, tokenized_prompts: torch.Tensor) -> torch.Tensor:
        position_embeddings = (
            self.clip_model.text_model.embeddings.position_embedding.weight.unsqueeze(0)
        )
        embedded_tokens = embedded_tokens + position_embeddings[:, : embedded_tokens.shape[1], :]

        encoder_outputs = self.clip_model.text_model.encoder(inputs_embeds=embedded_tokens)
        hidden_state = self.clip_model.text_model.final_layer_norm(encoder_outputs[0])

        # Pool at the EOT position (highest token id), as in CLIP.
        batch_indices = torch.arange(hidden_state.shape[0], device=tokenized_prompts.device)
        eot_indices = tokenized_prompts.to(torch.int).argmax(dim=-1)
        return hidden_state[batch_indices, eot_indices]


class PromptLearner(nn.Module):
    """Learnable context tokens with a frozen class-generic (aggregated) token.

    Args:
        class_names: e.g. ``["normal", "tumor"]``.
        clip_model: the frozen teacher VLM.
        processor: the matching ``CLIPProcessor``.
        num_context_tokens: K, the number of learnable tokens.
        agg_vector: ``[num_classes, embed_dim]`` aggregated template embeddings.
            Zeros are used if omitted (ablation without the class-generic token).
    """

    def __init__(
        self,
        class_names: list[str],
        clip_model: CLIPModel,
        processor: CLIPProcessor,
        num_context_tokens: int,
        agg_vector: torch.Tensor | None = None,
        embedding_dim: int = 512,
    ) -> None:
        super().__init__()
        self.class_names = list(class_names)
        self.num_context_tokens = int(num_context_tokens)
        self.num_classes = len(self.class_names)
        self.embedding_dim = embedding_dim
        self.dtype = clip_model.dtype

        if agg_vector is None:
            agg_vector = torch.zeros(
                (self.num_classes, self.embedding_dim),
                dtype=self.dtype,
                device=clip_model.device,
            )
        self.agg_vector = agg_vector
        self.text_features_base = agg_vector

        self.context_tokens = nn.Parameter(self._init_context_vectors(clip_model))
        self.prompt_prefix = " ".join(["X"] * self.num_context_tokens)

        token_embeddings = self._embed_prompt_skeleton(processor, clip_model, self.agg_vector)
        self._register_prefix_suffix(token_embeddings, clip_model)

    # ------------------------------------------------------------------ setup

    def _init_context_vectors(self, clip_model: CLIPModel) -> torch.Tensor:
        shape = (self.num_classes, self.num_context_tokens, self.embedding_dim)
        vectors = torch.empty(shape, dtype=self.dtype, device=clip_model.device)
        nn.init.normal_(vectors, std=0.02)
        return vectors

    def _embed_prompt_skeleton(
        self, processor: CLIPProcessor, clip_model: CLIPModel, agg_vector: torch.Tensor
    ) -> torch.Tensor:
        """Embed the skeleton 'X ... X C' and put the agg. vector in the C slot."""
        skeleton = [self.prompt_prefix + " C " for _ in self.class_names]
        self.tokenized_learnable_prompts = processor(
            text=skeleton, return_tensors="pt", padding=True
        ).input_ids.to(clip_model.device)

        with torch.no_grad():
            token_embeddings = clip_model.text_model.embeddings.token_embedding(
                self.tokenized_learnable_prompts
            ).type(self.dtype)
        token_embeddings[:, 1 + self.num_context_tokens, :] = agg_vector
        return token_embeddings

    def _register_prefix_suffix(self, token_embeddings: torch.Tensor, clip_model: CLIPModel) -> None:
        device = clip_model.device
        # [SOT]
        self.register_buffer("prefix_tokens", token_embeddings[:, :1, :].to(device))
        # class-generic token, [EOT] and padding
        self.register_buffer(
            "suffix_tokens", token_embeddings[:, 1 + self.num_context_tokens :, :].to(device)
        )

    # ---------------------------------------------------------------- forward

    def forward(self) -> torch.Tensor:
        context = self.context_tokens
        if context.dim() == 2:  # context shared across classes
            context = context.unsqueeze(0).expand(self.num_classes, -1, -1)
        context = context.to(self.prefix_tokens.device)
        return torch.cat([self.prefix_tokens, context, self.suffix_tokens], dim=1)

    @torch.no_grad()
    def text_features(self, clip_model: CLIPModel, normalize: bool = True) -> torch.Tensor:
        """Encode the current prompts into ``[num_classes, embed_dim]`` features."""
        features = TextEncoder(clip_model)(self(), self.tokenized_learnable_prompts)
        if normalize:
            features = features / features.norm(dim=-1, keepdim=True)
        return features


class LearnableAggPromptLearner(PromptLearner):
    """Ablation: the class-generic token is trained jointly with the context."""

    def __init__(
        self,
        class_names: list[str],
        clip_model: CLIPModel,
        processor: CLIPProcessor,
        num_context_tokens: int,
        agg_vector: torch.Tensor | None = None,
        embedding_dim: int = 512,
    ) -> None:
        super().__init__(
            class_names, clip_model, processor, num_context_tokens, agg_vector, embedding_dim
        )
        self.agg_vector = nn.Parameter(self.agg_vector.detach().clone())
        token_embeddings = self._embed_prompt_skeleton(processor, clip_model, self.agg_vector)
        self._register_prefix_suffix(token_embeddings, clip_model)


def build_prompt_learner(
    class_names: list[str],
    clip_model: CLIPModel,
    processor: CLIPProcessor,
    num_context_tokens: int,
    agg_vector: torch.Tensor | None = None,
    learnable_agg: bool = False,
    embedding_dim: int = 512,
) -> PromptLearner:
    cls = LearnableAggPromptLearner if learnable_agg else PromptLearner
    return cls(class_names, clip_model, processor, num_context_tokens, agg_vector, embedding_dim)


class PromptedCLIP(nn.Module):
    """Teacher VLM plus prompt learner, used only while *training* the prompts.

    ``forward`` returns ``(logits, drift)`` where ``drift`` is the KgCoOp-style
    penalty ``1 - cos(learned prompt, aggregated template prompt)`` that keeps the
    learned prompts anchored to the class-generic embedding.
    """

    def __init__(
        self,
        class_names: list[str],
        clip_model: CLIPModel,
        processor: CLIPProcessor,
        num_context_tokens: int = 4,
        agg_vector: torch.Tensor | None = None,
        learnable_agg: bool = False,
    ) -> None:
        super().__init__()
        self.prompt_learner = build_prompt_learner(
            class_names, clip_model, processor, num_context_tokens, agg_vector, learnable_agg
        )
        self.tokenized_prompts = self.prompt_learner.tokenized_learnable_prompts
        self.anchor_embedding = self.prompt_learner.text_features_base
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.clip_model = clip_model
        self.vision_model = clip_model.vision_model
        self.visual_projection = clip_model.visual_projection
        self.processor = processor

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pixel_values = clip_preprocess(images).to(self.logit_scale.device)

        vision_outputs = self.vision_model(pixel_values=pixel_values.type(self.dtype))
        image_features = self.visual_projection(vision_outputs.pooler_output)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        text_features = self.text_encoder(self.prompt_learner(), self.tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits = self.logit_scale.exp() * image_features @ text_features.t()

        anchor = self.anchor_embedding / self.anchor_embedding.norm(dim=-1, keepdim=True)
        drift = 1.0 - torch.mean(nn.CosineSimilarity(dim=1)(text_features, anchor))
        return logits, drift
