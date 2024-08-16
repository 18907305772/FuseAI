"""Code for DistillTrainer."""

import torch
from torch.nn.functional import (
    cross_entropy,
    kl_div,
    log_softmax,
    logsigmoid,
    margin_ranking_loss,
    one_hot,
    softmax,
)
from transformers import Seq2SeqTrainer
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

from .others import get_logger

logger = get_logger(__name__)


class DistillTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        if "base_target_dist" in inputs:
            base_target_dist = inputs.pop("base_target_dist")
            base_metric = inputs.pop("metric_ce")
        else:
            base_target_dist = None
            base_metric = None
        if "aligned_target_dist_0" in inputs:
            aligned_target_dist_0 = inputs.pop("aligned_target_dist_0")
            aligned_metric_0 = inputs.pop("metric_ce_aligned_0")
        else:
            aligned_target_dist_0 = None
            aligned_metric_0 = None
        if "aligned_target_dist_1" in inputs:
            aligned_target_dist_1 = inputs.pop("aligned_target_dist_1")
            aligned_metric_1 = inputs.pop("metric_ce_aligned_1")
        else:
            aligned_target_dist_1 = None
            aligned_metric_1 = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if (
                unwrap_model(model)._get_name()
                in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values()
            ):
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if self.args.do_distill:
            batch_size, seq_len, vocab_size = (
                outputs["logits"].size(0),
                outputs["logits"].size(1),
                outputs["logits"].size(2),
            )
            align_reward_0 = (
                (
                    1 / torch.exp(torch.tensor(aligned_metric_0, dtype=torch.bfloat16))
                ).to(loss.device)
                if aligned_target_dist_0 is not None
                else None
            )
            align_reward_1 = (
                (
                    1 / torch.exp(torch.tensor(aligned_metric_1, dtype=torch.bfloat16))
                ).to(loss.device)
                if aligned_target_dist_1 is not None
                else None
            )
            base_reward = (
                (1 / torch.exp(torch.tensor(base_metric, dtype=torch.bfloat16))).to(
                    loss.device
                )
                if base_target_dist is not None
                else None
            )

            if self.args.distill_greater_as_gt is True:
                if base_target_dist is None:
                    align_reward_0_expanded = (
                        align_reward_0.unsqueeze(-1)
                        .unsqueeze(-1)
                        .expand_as(aligned_target_dist_0)
                        if aligned_target_dist_0 is not None
                        else None
                    )
                    align_reward_1_expanded = (
                        align_reward_1.unsqueeze(-1)
                        .unsqueeze(-1)
                        .expand_as(aligned_target_dist_1)
                        if aligned_target_dist_1 is not None
                        else None
                    )
                    if (
                        aligned_target_dist_0 is not None
                        and aligned_target_dist_1 is not None
                    ):
                        target_dist = torch.where(
                            align_reward_0_expanded > align_reward_1_expanded,
                            aligned_target_dist_0,
                            aligned_target_dist_1,
                        )
                    elif aligned_target_dist_0 is not None:
                        target_dist = aligned_target_dist_0
                    elif aligned_target_dist_1 is not None:
                        target_dist = aligned_target_dist_1
                    else:
                        raise ValueError
                    if self.args.distill_loss_type == "ce":
                        loss_lm = cross_entropy(
                            input=outputs["logits"].view(-1, vocab_size),
                            target=target_dist.view(-1, vocab_size),
                            reduction="none",
                        ).view(batch_size, -1)
                    elif self.args.distill_loss_type == "kl":
                        loss_lm = kl_div(
                            input=log_softmax(outputs["logits"], dim=-1),
                            target=target_dist,
                            log_target=False,
                            reduction="none",
                        ).sum(dim=-1)
                    loss_lm = (loss_lm * inputs["attention_mask"]).sum() / inputs[
                        "attention_mask"
                    ].sum()
                    if self.args.distill_greater_as_gt_type == "hard":
                        loss = (
                            self.args.lm_loss_weight * loss
                            + (1.0 - self.args.lm_loss_weight) * loss_lm
                        )
                    elif self.args.distill_greater_as_gt_type == "hard_and_decay":
                        decay_lm_loss_weight = self.args.lm_loss_weight + (
                            1.0 - self.args.lm_loss_weight
                        ) * (self.state.global_step / self.state.max_steps)
                        loss = (
                            decay_lm_loss_weight * loss
                            + (1.0 - decay_lm_loss_weight) * loss_lm
                        )
                    elif self.args.distill_greater_as_gt_type == "soft":
                        max_reward = torch.max(
                            torch.stack([align_reward_0, align_reward_1], dim=-1),
                            dim=-1,
                        )[0]
                        assert batch_size == 1
                        loss = (1.0 - max_reward[0]) * loss + max_reward[0] * loss_lm
                    else:
                        raise NotImplementedError
                else:
                    base_reward_expanded = (
                        base_reward.unsqueeze(-1)
                        .unsqueeze(-1)
                        .expand_as(base_target_dist)
                        if base_target_dist is not None
                        else None
                    )
                    align_reward_0_expanded = (
                        align_reward_0.unsqueeze(-1)
                        .unsqueeze(-1)
                        .expand_as(aligned_target_dist_0)
                        if aligned_target_dist_0 is not None
                        else None
                    )
                    align_reward_1_expanded = (
                        align_reward_1.unsqueeze(-1)
                        .unsqueeze(-1)
                        .expand_as(aligned_target_dist_1)
                        if aligned_target_dist_1 is not None
                        else None
                    )
                    target_dist_list = []
                    reward_list = []
                    if base_target_dist is not None:
                        target_dist_list.append(base_target_dist)
                        reward_list.append(base_reward_expanded)
                    if aligned_target_dist_0 is not None:
                        target_dist_list.append(aligned_target_dist_0)
                        reward_list.append(align_reward_0_expanded)
                    if aligned_target_dist_1 is not None:
                        target_dist_list.append(aligned_target_dist_1)
                        reward_list.append(align_reward_1_expanded)
                    stacked_dists = torch.stack(target_dist_list, dim=-1)
                    stacked_rewards = torch.stack(reward_list, dim=-1)
                    max_reward_indices = torch.argmax(
                        stacked_rewards, dim=-1, keepdim=True
                    )
                    target_dist = torch.gather(
                        stacked_dists, -1, max_reward_indices
                    ).squeeze(-1)
                    if self.args.distill_loss_type == "ce":
                        loss_lm = cross_entropy(
                            input=outputs["logits"].view(-1, vocab_size),
                            target=target_dist.view(-1, vocab_size),
                            reduction="none",
                        ).view(batch_size, -1)
                    elif self.args.distill_loss_type == "kl":
                        loss_lm = kl_div(
                            input=log_softmax(outputs["logits"], dim=-1),
                            target=target_dist,
                            log_target=False,
                            reduction="none",
                        ).sum(dim=-1)
                    loss_lm = (loss_lm * inputs["attention_mask"]).sum() / inputs[
                        "attention_mask"
                    ].sum()
                    if self.args.distill_greater_as_gt_type == "hard":
                        loss = (
                            self.args.lm_loss_weight * loss
                            + (1.0 - self.args.lm_loss_weight) * loss_lm
                        )
                    elif self.args.distill_greater_as_gt_type == "hard_and_decay":
                        decay_lm_loss_weight = self.args.lm_loss_weight + (
                            1.0 - self.args.lm_loss_weight
                        ) * (self.state.global_step / self.state.max_steps)
                        loss = (
                            decay_lm_loss_weight * loss
                            + (1.0 - decay_lm_loss_weight) * loss_lm
                        )
                    elif self.args.distill_greater_as_gt_type == "soft":
                        max_reward = torch.max(
                            torch.stack(
                                [base_reward, align_reward_0, align_reward_1], dim=-1
                            ),
                            dim=-1,
                        )[0]
                        assert batch_size == 1
                        loss = (1.0 - max_reward[0]) * loss + max_reward[0] * loss_lm
                    else:
                        raise NotImplementedError
            elif self.args.distill_weighted_as_gt is True:
                if (
                    base_target_dist is not None
                    and aligned_target_dist_0 is not None
                    and aligned_target_dist_1 is not None
                ):
                    weights = torch.stack(
                        [base_reward, align_reward_0, align_reward_1], dim=1
                    )
                    normalized_weights = torch.softmax(weights, dim=1)
                    weighted_label = (
                        normalized_weights[:, 0].unsqueeze(1).unsqueeze(2)
                        * base_target_dist
                        + normalized_weights[:, 1].unsqueeze(1).unsqueeze(2)
                        * aligned_target_dist_0
                        + normalized_weights[:, 2].unsqueeze(1).unsqueeze(2)
                        * aligned_target_dist_1
                    )
                elif (
                    aligned_target_dist_0 is not None
                    and aligned_target_dist_1 is not None
                ):
                    weights = torch.stack([align_reward_0, align_reward_1], dim=1)
                    normalized_weights = torch.softmax(weights, dim=1)
                    weighted_label = (
                        normalized_weights[:, 0].unsqueeze(1).unsqueeze(2)
                        * aligned_target_dist_0
                        + normalized_weights[:, 1].unsqueeze(1).unsqueeze(2)
                        * aligned_target_dist_1
                    )
                elif base_target_dist is not None and aligned_target_dist_0 is not None:
                    weights = torch.stack([base_reward, align_reward_0], dim=1)
                    normalized_weights = torch.softmax(weights, dim=1)
                    weighted_label = (
                        normalized_weights[:, 0].unsqueeze(1).unsqueeze(2)
                        * base_target_dist
                        + normalized_weights[:, 1].unsqueeze(1).unsqueeze(2)
                        * aligned_target_dist_0
                    )
                elif base_target_dist is not None and aligned_target_dist_1 is not None:
                    weights = torch.stack([base_reward, align_reward_1], dim=1)
                    normalized_weights = torch.softmax(weights, dim=1)
                    weighted_label = (
                        normalized_weights[:, 0].unsqueeze(1).unsqueeze(2)
                        * base_target_dist
                        + normalized_weights[:, 1].unsqueeze(1).unsqueeze(2)
                        * aligned_target_dist_1
                    )
                else:
                    raise ValueError
                if self.args.distill_loss_type == "ce":
                    loss_lm = cross_entropy(
                        input=outputs["logits"].view(-1, vocab_size),
                        target=weighted_label.view(-1, vocab_size),
                        reduction="none",
                    ).view(batch_size, -1)
                elif self.args.distill_loss_type == "kl":
                    loss_lm = kl_div(
                        input=log_softmax(outputs["logits"], dim=-1),
                        target=weighted_label,
                        log_target=False,
                        reduction="none",
                    ).sum(dim=-1)
                else:
                    raise NotImplementedError
                loss_lm = (loss_lm * inputs["attention_mask"]).sum() / inputs[
                    "attention_mask"
                ].sum()
                if self.args.distill_weighted_as_gt_type == "hard":
                    loss = (
                        self.args.lm_loss_weight * loss
                        + (1.0 - self.args.lm_loss_weight) * loss_lm
                    )
                elif self.args.distill_weighted_as_gt_type == "hard_and_decay":
                    decay_lm_loss_weight = self.args.lm_loss_weight + (
                        1.0 - self.args.lm_loss_weight
                    ) * (self.state.global_step / self.state.max_steps)
                    loss = (
                        decay_lm_loss_weight * loss
                        + (1.0 - decay_lm_loss_weight) * loss_lm
                    )
                elif self.args.distill_weighted_as_gt_type == "soft":
                    mean_reward = weights.mean(dim=1)
                    assert batch_size == 1
                    loss = (1.0 - mean_reward[0]) * loss + mean_reward[0] * loss_lm
                else:
                    raise NotImplementedError
            else:
                loss = self.args.lm_loss_weight * loss

        return (loss, outputs) if return_outputs else loss
