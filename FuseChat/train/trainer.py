import torch
from torch.nn.functional import softmax, kl_div, log_softmax, cross_entropy, margin_ranking_loss, logsigmoid, one_hot
from transformers import Seq2SeqTrainer
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.trainer_pt_utils import LabelSmoother
IGNORE_TOKEN_ID = LabelSmoother.ignore_index

class SFTTrainer(Seq2SeqTrainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
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
        return (loss, outputs) if return_outputs else loss

class FuseTrainer(Seq2SeqTrainer):

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
        if "fused_target_dist" in inputs:
            fuse_target_dist = inputs.pop("fused_target_dist")
            fuse_metric = inputs.pop("fused_metric_ce")
        else:
            fuse_target_dist = None
            fuse_metric = None

        outputs = model(**inputs)
        # Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
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

        batch_size, seq_len, vocab_size = outputs["logits"].size(0), outputs["logits"].size(1), outputs["logits"].size(2)
        values_to_add = torch.full((batch_size, 1), IGNORE_TOKEN_ID, device=inputs["labels"].device)
        loss_mask = torch.cat((inputs["labels"][..., 1:], values_to_add), dim=-1).ne(IGNORE_TOKEN_ID)  # TODO: Fixed Bug.
        fuse_reward = (1 / torch.exp(torch.tensor(fuse_metric, dtype=torch.bfloat16))).to(loss.device) if fuse_target_dist is not None else None
        base_reward = (1 / torch.exp(torch.tensor(base_metric, dtype=torch.bfloat16))).to(loss.device) if base_target_dist is not None else None

        if base_target_dist is None:
            if self.args.fuse_loss_type == "ce":
                loss_lm = cross_entropy(input=outputs["logits"].view(-1, vocab_size),
                                        target=fuse_target_dist.view(-1, vocab_size),
                                        reduction="none").view(batch_size, -1)  # (bs, seq_len)
            elif self.args.fuse_loss_type == "kl":
                loss_lm = kl_div(input=log_softmax(outputs["logits"], dim=-1),
                                 target=fuse_target_dist,
                                 log_target=False,
                                 reduction="none").sum(dim=-1)  # (bs, seq_len)
            loss_lm = (loss_lm * loss_mask).sum() / loss_mask.sum()
            loss = self.args.lm_loss_weight * loss + (1.0 - self.args.lm_loss_weight) * loss_lm
        else:
            base_reward_expanded = base_reward.unsqueeze(-1).unsqueeze(-1).expand_as(base_target_dist) if base_target_dist is not None else None
            fuse_reward_expanded = fuse_reward.unsqueeze(-1).unsqueeze(-1).expand_as(fuse_target_dist) if fuse_target_dist is not None else None
            target_dist_list = []
            reward_list = []
            if base_target_dist is not None:
                target_dist_list.append(base_target_dist)
                reward_list.append(base_reward_expanded)
            if fuse_target_dist is not None:
                target_dist_list.append(fuse_target_dist)
                reward_list.append(fuse_reward_expanded)
            stacked_dists = torch.stack(target_dist_list, dim=-1)
            stacked_rewards = torch.stack(reward_list, dim=-1)
            max_reward_indices = torch.argmax(stacked_rewards, dim=-1, keepdim=True)
            target_dist = torch.gather(stacked_dists, -1, max_reward_indices).squeeze(-1)
            if self.args.fuse_loss_type == "ce":
                loss_lm = cross_entropy(input=outputs["logits"].view(-1, vocab_size),
                                        target=target_dist.view(-1, vocab_size),
                                        reduction="none").view(batch_size, -1)  # (bs, seq_len)
            elif self.args.fuse_loss_type == "kl":
                loss_lm = kl_div(input=log_softmax(outputs["logits"], dim=-1),
                                 target=target_dist,
                                 log_target=False,
                                 reduction="none").sum(dim=-1)  # (bs, seq_len)
            loss_lm = (loss_lm * loss_mask).sum() / loss_mask.sum()
            loss = self.args.lm_loss_weight * loss + (1.0 - self.args.lm_loss_weight) * loss_lm

        return (loss, outputs) if return_outputs else loss