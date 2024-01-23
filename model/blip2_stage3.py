import os
from typing import Any, Dict
import torch
from model.blip2_opt import Blip2OPT
from model.blip2_llama import Blip2Llama
import pytorch_lightning as pl
from torch import optim
import json
import torch.distributed as dist
from peft import LoraConfig, TaskType
from model.help_funcs import caption_evaluate, AttrDict
from model.llama_flash_attention import replace_flash_attn_with_llama_attn, replace_llama_attn_with_flash_attn
import pickle
import math


class LinearWarmupCosineLRScheduler:
    def __init__(
        self,
        optimizer,
        max_step,
        min_lr,
        init_lr,
        warmup_steps=0,
        warmup_start_lr=-1,
        **kwargs
    ):
        self.optimizer = optimizer

        self.max_step = max_step
        self.min_lr = min_lr

        self.init_lr = init_lr
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr if warmup_start_lr >= 0 else init_lr

    def step(self, cur_step):
        # assuming the warmup iters less than one epoch
        if cur_step <= self.warmup_steps:
            warmup_lr_schedule(
                step=cur_step,
                optimizer=self.optimizer,
                max_step=self.warmup_steps,
                init_lr=self.warmup_start_lr,
                max_lr=self.init_lr,
            )
        else:
            cosine_lr_schedule(
                step=cur_step,
                optimizer=self.optimizer,
                max_step=self.max_step,
                init_lr=self.init_lr,
                min_lr=self.min_lr,
            )


def cosine_lr_schedule(optimizer, step, max_step, init_lr, min_lr):
    """Decay the learning rate"""
    lr = (init_lr - min_lr) * 0.5 * (
        1.0 + math.cos(math.pi * step / max_step)
    ) + min_lr
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def warmup_lr_schedule(optimizer, step, max_step, init_lr, max_lr):
    """Warmup the learning rate"""
    lr = min(max_lr, init_lr + (max_lr - init_lr) * step / max(max_step, 1))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def load_ignore_unexpected(model, state_dict):
    keys = set(model.state_dict().keys())
    state_dict = {k: v for k, v in state_dict.items() if k in keys}

    ## try to print keys that are not included
    model.load_state_dict(state_dict, strict=True)


def get_module_state_dict(state_dict, module_name):
    module_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(module_name):
            key = key[len(module_name) + 1:]
            if key == '':
                return value
            module_state_dict[key] = value
    return module_state_dict


class Blip2Stage3(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        if isinstance(args, dict):
            args = AttrDict(**args)

        self.args = args
        if not hasattr(args, 'do_sample'):
            args.do_sample = False
        self.do_sample = args.do_sample
        self.num_beams = args.num_beams
        self.max_len = args.max_len
        self.min_len = args.min_len
        self.llm_tune = args.llm_tune
        self.blip2opt = Blip2Llama(args.bert_name, args.gin_num_layers, args.gin_hidden_dim, args.drop_ratio,
                                   args.tune_gnn, args.num_query_token, args.cross_attention_freq, args.llm_tune,
                                   args.llm_model, args)

        self.tokenizer = self.blip2opt.init_tokenizer()
        self.enable_flash = args.enable_flash
        self.save_hyperparameters(args)
        self.test_step_outputs = []

    def configure_optimizers(self):
        self.trainer.fit_loop.setup_data()
        max_step = self.args.max_steps
        warmup_steps = self.args.warmup_steps
        optimizer = optim.AdamW(self.parameters(), lr=self.args.init_lr, weight_decay=self.args.weight_decay)
        if self.args.scheduler == 'linear_warmup_cosine_lr':
            self.scheduler = LinearWarmupCosineLRScheduler(optimizer, max_step, self.args.min_lr, self.args.init_lr, warmup_steps, self.args.warmup_lr)
        elif self.args.scheduler == 'None':
            self.scheduler = None
        else:
            raise NotImplementedError()
        return optimizer

    def on_validation_epoch_end(self):
        if self.enable_flash:
            replace_llama_attn_with_flash_attn()

    def on_validation_epoch_start(self) -> None:
        if self.enable_flash:
            replace_flash_attn_with_llama_attn()

    def on_test_epoch_start(self) -> None:
        if self.enable_flash:
            replace_flash_attn_with_llama_attn()

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint.pop('optimizer_states')
        to_be_removed = []
        for key, value in checkpoint['state_dict'].items():
            try:
                if not self.get_parameter(key).requires_grad:
                    to_be_removed.append(key)
            except AttributeError:
                to_be_removed.append(key)
        for key in to_be_removed:
            checkpoint['state_dict'].pop(key)

    def training_step(self, batch, batch_idx):
        if self.scheduler:
            self.scheduler.step(self.trainer.global_step)

        batch_size = batch[-1].input_ids.size(0)
        ###============== Overall Loss ===================###
        loss = self.blip2opt(batch)
        self.log("molecule loss", float(loss['loss']), batch_size=batch_size, sync_dist=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], batch_size=batch_size, sync_dist=True)
        return loss['loss']

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):

        batch_size = batch[-1].input_ids.size(0)
        loss = self.blip2opt(batch)
        ###============== Overall Loss ===================###
        self.log("val molecule loss", float(loss['loss']), batch_size=batch_size, sync_dist=True)
        return loss['loss']

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        graphs, prompt_tokens, texts = batch
        if texts['task_type'][-1] in ["HOMO",
                                      "LUMO",
                                      "HOMO-LUMO Gap",
                                      "SCF Energy",
                                      "Molecular Weight",
                                      "LogP",
                                      "Topological Polar Surface Area",
                                      "Complexity",
                                      ]:
            max_new_tokens = 64
            min_new_tokens = 16
            length_penalty = 0.8
        elif texts['task_type'][-1] in ["Description"]:
            max_new_tokens = 64
            min_new_tokens = 16
            length_penalty = 0.8
        elif texts['task_type'][-1] in ["Caption"]:
            max_new_tokens = 128
            min_new_tokens = 16
            length_penalty = 0.8
        else:
            raise NotImplementedError

        ###============== Captioning Results ===================###
        samples = {'graphs': graphs, 'prompt_tokens': prompt_tokens}
        predictions = self.blip2opt.generate(
            samples,
            do_sample=self.args.do_sample,
            num_beams=self.args.num_beams,
            max_length=self.args.max_len,
            min_length=self.args.min_len,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            length_penalty=length_penalty,
        )
        self.test_step_outputs.append((predictions, texts['targets'], texts['task_type']))
        return predictions, texts['targets'], texts['task_type']

    def on_test_epoch_end(self):
        outputs = self.test_step_outputs

        list_predictions, list_targets, list_task_type = zip(*outputs)
        predictions = [i for ii in list_predictions for i in ii]
        targets = [i for ii in list_targets for i in ii]
        task_types = [i for ii in list_task_type for i in ii]

        all_predictions = [None for _ in range(self.trainer.world_size)]
        all_targets = [None for _ in range(self.trainer.world_size)]
        all_task_types = [None for _ in range(self.trainer.world_size)]

        dist.all_gather_object(all_predictions, predictions)
        dist.all_gather_object(all_targets, targets)
        dist.all_gather_object(all_task_types, task_types)

        if self.global_rank == 0:
            self.save_predictions(all_predictions, all_targets, all_task_types)

    def save_predictions(self, predictions, targets, task_types):
        assert len(predictions) == len(targets) == len(task_types)
        with open(os.path.join(self.logger.log_dir, f'predictions.txt'), 'w', encoding='utf8') as f:
            for p, t, task in zip(predictions, targets, task_types):
                line = {'prediction': p, 'target': t, "task_type": task}
                f.write(json.dumps(line, ensure_ascii=False) + '\n')

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("3D-MLM")
        # GIN
        parser.add_argument('--gin_hidden_dim', type=int, default=300)
        parser.add_argument('--gin_num_layers', type=int, default=5)
        parser.add_argument('--drop_ratio', type=float, default=0.0)
        parser.add_argument('--tune_gnn', action='store_true', default=False)
        # Bert
        parser.add_argument('--bert_hidden_dim', type=int, default=768, help='')
        parser.add_argument('--bert_name', type=str, default='scibert')
        parser.add_argument('--cross_attention_freq', type=int, default=2)
        parser.add_argument('--num_query_token', type=int, default=8)
        # OPT
        parser.add_argument('--llm_model', type=str, default="all_checkpoints/llama-2-7b-hf")
        parser.add_argument('--num_beams', type=int, default=5)
        parser.add_argument('--do_sample', action='store_true', default=False)
        parser.add_argument('--max_len', type=int, default=64)
        parser.add_argument('--min_len', type=int, default=4)

        parser.add_argument('--llm_tune', type=str, default='lora')
        parser.add_argument('--peft_config', type=str, default=None)
        parser.add_argument('--peft_dir', type=str, default='')

        parser.add_argument('--every_n_train_steps', type=int, default=5000)
        # parser.add_argument('--save_every_n_epochs', type=int, default=1)
        ## quantization
        parser.add_argument('--load_in_8bit', action='store_true', default=False)

        ## lora config
        parser.add_argument('--lora_r', type=int, default=8)
        parser.add_argument('--lora_alpha', type=int, default=32)
        parser.add_argument('--lora_dropout', type=int, default=0.1)

        # optimization
        parser.add_argument('--weight_decay', type=float, default=0.05, help='optimizer weight decay')
        parser.add_argument('--init_lr', type=float, default=1e-5, help='optimizer init learning rate')
        parser.add_argument('--min_lr', type=float, default=1e-8, help='optimizer min learning rate')
        parser.add_argument('--warmup_lr', type=float, default=1e-6, help='optimizer warmup learning rate')
        parser.add_argument('--warmup_steps', type=int, default=1000, help='optimizer warmup steps')
        parser.add_argument('--scheduler', type=str, default='linear_warmup_cosine_lr', help='type of scheduler')
        parser.add_argument('--stage2_path', type=str, default='')
        parser.add_argument('--stage3_path', type=str, default='')
        parser.add_argument('--init_checkpoint', type=str, default='')
        return parent_parser


