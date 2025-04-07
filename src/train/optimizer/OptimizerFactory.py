from typing import Iterable

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR, OneCycleLR
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

class OptimizerFactory:
    @staticmethod
    def create_sgd_optimizer_with_nesterov_momentum(parameters: Iterable[torch.Tensor], learning_rate=2e-5,
                                                    momentum=0.9) -> torch.optim.Optimizer:
        return torch.optim.SGD(parameters, lr=learning_rate, momentum=momentum, nesterov=True, weight_decay=0.0001)

    @staticmethod
    def create_adamw_optimizer(model) -> torch.optim.AdamW:
        """
        alchemy optimizer
        """
        no_decay = ['bias', 'LayerNorm.weight']
        bert_params = [(n, p) for n, p in model.text_model.named_parameters()]
        vit_params = [(n, p) for n, p in model.visual_model.named_parameters()]
        fusion_params = [(n, p) for n, p in model.named_parameters() if
                         any(x in n for x in ['fusion_layer', 'projection_layer', 'text_projection_layer'])]
        bilstm_params = [(n, p) for n, p in model.named_parameters() if 'bilstm' in n]
        crf_params = [(n, p) for n, p in model.named_parameters() if 'crf' in n]
        other_params = [(n, p) for n, p in model.named_parameters() if
                        not any(x in n for x in ['text_model', 'visual_model', 'fusion_layer',
                                                 'projection_layer', 'text_projection_layer', 'bilstm', 'crf'])]
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in bert_params if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.01,
                'lr': 8e-6
            },
            {
                'params': [p for n, p in bert_params if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                'lr': 8e-6
            },
            # ViT parameters
            {
                'params': [p for n, p in vit_params if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.01,
                'lr': 8e-5
            },
            {
                'params': [p for n, p in vit_params if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                'lr': 8e-5
            },
            # Fusion layer
            {
                'params': [p for n, p in fusion_params if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.01,
                'lr': 5e-4
            },
            {
                'params': [p for n, p in fusion_params if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                'lr': 5e-4
            },
            # BiLSTM parameters
            {
                'params': [p for n, p in bilstm_params if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.01,
                'lr': 3e-4  # Modified learning rate
            },
            {
                'params': [p for n, p in bilstm_params if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                'lr': 3e-4
            },
            # CRF parameters
            {
                'params': [p for n, p in crf_params],
                'weight_decay': 0.0,  # No weight decay for CRF
                'lr': 2e-4  # Higher learning rate for CRF
            },
            # Other parameters
            {
                'params': [p for n, p in other_params if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.01,
                'lr': 1e-5
            },
            {
                'params': [p for n, p in other_params if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                'lr': 1e-5
            }
        ]
        return torch.optim.AdamW(optimizer_grouped_parameters, weight_decay=0.01, eps=1e-9, betas=(0.9, 0.98))

    @staticmethod
    def create_linear_scheduler(optimizer, training_steps, warmup_steps):
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=training_steps
        )

    @staticmethod
    def create_warm_cosine_scheduler(optimizer):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=5,
            T_mult=2,
            eta_min=1e-6
        )
        return scheduler

    @staticmethod
    def create_cosine_scheduler(optimizer, training_steps):
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(training_steps * 0.15),  # 15% warmup
            num_training_steps=training_steps
        )

    @staticmethod
    def create_plateau_scheduler(optimizer):
        return ReduceLROnPlateau(optimizer, 'max', patience=1, threshold=0.01)

    @staticmethod
    def create_cyclic_scheduler(optimizer):
        return CyclicLR(
            optimizer,
            base_lr=1e-7,
            max_lr=1e-6,
            step_size_up=300,
            mode='triangular2',
            cycle_momentum=False)

    @staticmethod
    def create_one_cycle_scheduler(optimizer, num_of_training_steps):
        return OneCycleLR(
            optimizer,
            max_lr=5e-6,
            total_steps=num_of_training_steps,
            pct_start=0.15,
            anneal_strategy='cos',
            div_factor=25,
            final_div_factor=1000
        )
