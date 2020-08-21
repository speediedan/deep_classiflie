import logging
import os
from itertools import chain
from typing import MutableMapping, Union, Tuple

import torch

import utils.constants as constants
from models.custom_albert_module import AlbertModel
from training.training_utils import load_old_ckpt, smoothed_label_bce

logger = logging.getLogger(constants.APP_NAME)


class DeepClassiflie(torch.nn.Module):
    def __init__(self, config: MutableMapping) -> None:
        super().__init__()
        self.config = config
        base_ckpt = os.path.join(self.config.experiment.dirs.model_cache_dir, self.config.model.base_ckpt)
        self.albert = load_old_ckpt(AlbertModel(self.config), base_ckpt,
                                    strip_prefix=self.config.model.strip_prefix, mode='train')
        self.config.model.num_labels = 1
        self.stmt_pooled_embed = None
        self.dropout = torch.nn.Dropout(self.config.model.hidden_dropout_prob)
        self.ctxt_embed = torch.nn.Embedding(2, self.config.model.hidden_size)
        self.ctxt_lnorm = torch.nn.LayerNorm(self.config.model.hidden_size, eps=config.model.layer_norm_eps)
        self.ctxt_fc = torch.nn.Linear(self.config.model.hidden_size, self.config.model.hidden_size)
        self.relu = torch.nn.ReLU()
        self.classifier = torch.nn.Linear(self.config.model.hidden_size, self.config.model.num_labels)
        self.sigmoid = torch.nn.Sigmoid()
        self.interpretation_mode = False
        param_gens = [l.parameters() for l in [self.ctxt_embed, self.albert]]
        # assume initially frozen for subsequent fine-tuning
        for param in chain(*param_gens):
            param.requires_grad = False
        uninit_layers = [self.classifier, self.ctxt_fc, self.ctxt_lnorm, self.ctxt_embed]
        for m in uninit_layers:
            m.apply(self._init_weights)

    # changing labels position to workaround add_graph tracing limitations
    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None, attention_mask: torch.Tensor = None,
                token_type_ids: torch.Tensor = None, position_ids: torch.Tensor = None,
                head_mask: torch.Tensor = None, ctxt_type: torch.Tensor = None) -> Union[torch.Tensor, Tuple]:
        outputs = self.albert(input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              position_ids=position_ids,
                              head_mask=head_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        ctxt_type = torch.unsqueeze(ctxt_type.to(dtype=torch.long), -1)
        ctxt_embeds = self.ctxt_embed(ctxt_type)
        ctxt_embeds = self.ctxt_lnorm(ctxt_embeds)
        ctxt_embeds = self.dropout(ctxt_embeds)
        ctxt_in = torch.add(pooled_output, torch.squeeze(ctxt_embeds))
        ctxt_output = self.relu(self.ctxt_fc(ctxt_in))
        ctxt_output = self.dropout(ctxt_output)
        logits = self.classifier(ctxt_output)
        # don't detach probs if we're in interpretation mode since we'll be using them
        probs = self.sigmoid(logits)
        if self.interpretation_mode:
            self.stmt_pooled_embed = ctxt_output.detach()
            outputs = probs
            return outputs
        else:
            if self.config.trainer.label_smoothing_enabled:
                loss = smoothed_label_bce(probs.view(-1), labels.view(-1), smoothing=self.config.trainer.smoothing)
            else:
                loss_fct = torch.nn.BCELoss()
                loss = loss_fct(probs.view(-1), labels.view(-1))
            outputs = (loss,) + (probs.detach().cpu(),)
            return outputs  # (loss), (probs)

    def _init_weights(self, module: torch.nn.Module):
        """ Initialize the weights """
        if isinstance(module, torch.nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.model.ctxt_init_range)
        elif isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.model.initializer_range)
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def set_interpret_mode(self):
        self.interpretation_mode = True
