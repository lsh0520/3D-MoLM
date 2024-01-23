import os
from typing import Any, Dict
import torch
from model.blip2_opt import Blip2OPT
from model.blip2_llama import Blip2Llama
import pytorch_lightning as pl
from torch import optim
from lavis.common.optims import LinearWarmupCosineLRScheduler, LinearWarmupStepLRScheduler
import json
import torch.distributed as dist
from peft import LoraConfig, TaskType
from model.help_funcs import caption_evaluate, AttrDict
from model.llama_flash_attention import replace_flash_attn_with_llama_attn, replace_llama_attn_with_flash_attn
import pickle


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


class Blip2Stage2(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        if isinstance(args, dict):
            args = AttrDict(**args)

        self.args = args
        if not hasattr(args, 'do_sample'):
            args.do_sample = False
        self.caption_eval_epoch = args.caption_eval_epoch
        self.llm_tune = args.llm_tune
        self.blip2opt = Blip2Llama(args.bert_name,
                                   args.gin_num_layers,
                                   args.gin_hidden_dim,
                                   args.drop_ratio,
                                   args.tune_gnn,
                                   args.num_query_token,
                                   args.cross_attention_freq,
                                   args.llm_tune,
                                   args.llm_model, 
                                   args)
        self.tokenizer = self.blip2opt.init_tokenizer()
        self.enable_flash = args.enable_flash
        self.save_hyperparameters(args)

        self.test_step_outputs = []

    def load_from_stage1_checkpoint(self, path):
        print('loading from stage1 checkpoint')
        ckpt = torch.load(path, map_location='cpu')
        state_dict = ckpt['state_dict']
        state_dict = {k.split('blip2qformer.')[1]: v for k, v in state_dict.items()}
        missing_keys, unexpected_keys = self.blip2opt.load_state_dict(state_dict, strict=False)
        print('missing keys')
        print(missing_keys)
        print('unexpected keys')
        print(unexpected_keys)
        return self

    def configure_optimizers(self):
        self.trainer.fit_loop.setup_data()
        warmup_steps = min(len(self.trainer.train_dataloader), self.args.warmup_steps)
        optimizer = optim.AdamW(self.parameters(), lr=self.args.init_lr, weight_decay=self.args.weight_decay)
        if self.args.scheduler == 'linear_warmup_cosine_lr':
            self.scheduler = LinearWarmupCosineLRScheduler(optimizer, self.args.max_epochs, self.args.min_lr, self.args.init_lr, warmup_steps, self.args.warmup_lr)
        elif self.args.scheduler == 'linear_warmup_step_lr':
            self.scheduler = LinearWarmupStepLRScheduler(optimizer, self.args.max_epochs, self.args.min_lr, self.args.init_lr, self.args.lr_decay_rate, self.args.warmup_lr, warmup_steps)
        elif self.args.scheduler == 'None':
            self.scheduler = None
        else:
            raise NotImplementedError()
        return optimizer

    def on_validation_epoch_end(self):
        if self.enable_flash:
            replace_llama_attn_with_flash_attn()

    # def on_train_epoch_start(self) -> None:
    #     if self.enable_flash:
    #         replace_flash_attn_with_llama_attn()

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

    def save_predictions(self, predictions, targets):
        assert len(predictions) == len(targets)
        with open(os.path.join(self.logger.log_dir, 'predictions.txt'), 'w', encoding='utf8') as f:
            for p, t in zip(predictions, targets):
                line = {'prediction': p, 'target': t}
                f.write(json.dumps(line, ensure_ascii=True) + '\n')

    def training_step(self, batch, batch_idx):
        if self.scheduler:
            self.scheduler.step(self.trainer.current_epoch, self.trainer.global_step)

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
        ###============== Captioning Results ===================###
        samples = {'graphs': graphs, 'prompt_tokens': prompt_tokens}
        predictions = self.blip2opt.generate(
            samples,
            do_sample=self.args.do_sample,
            num_beams=self.args.num_beams,
            max_length=self.args.max_len,
            min_length=self.args.min_len,
            max_new_tokens=self.args.max_new_tokens,
            min_new_tokens=self.args.min_new_tokens,
            length_penalty=self.args.length_penalty,
            repetition_penalty=self.args.repetition_penalty,
        )
        self.test_step_outputs.append((predictions, texts['targets']))
        return predictions, texts['targets']

    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        list_predictions, list_targets = zip(*outputs)
        predictions = [i for ii in list_predictions for i in ii]
        targets = [i for ii in list_targets for i in ii]

        all_predictions = [None for _ in range(self.trainer.world_size)]
        all_targets = [None for _ in range(self.trainer.world_size)]

        dist.all_gather_object(all_predictions, predictions)
        dist.all_gather_object(all_targets, targets)
        if self.global_rank == 0:
            all_predictions = [i for ii in all_predictions for i in ii]
            all_targets = [i for ii in all_targets for i in ii]
            self.save_predictions(all_predictions, all_targets)
            ## fixme: I am not sure if the max length is the same as previous experiments
            bleu2, bleu4, rouge_1, rouge_2, rouge_l, meteor_score = \
                caption_evaluate(all_predictions, all_targets, self.tokenizer, self.args.max_new_tokens)
            self.log("bleu2", bleu2, sync_dist=False)
            self.log("bleu4", bleu4, sync_dist=False)
            self.log("rouge_1", rouge_1, sync_dist=False)
            self.log("rouge_2", rouge_2, sync_dist=False)
            self.log("rouge_l", rouge_l, sync_dist=False)
            self.log("meteor_score", meteor_score, sync_dist=False)


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("GINSimclr")
        # train mode
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
        parser.add_argument('--max_len', type=int, default=128)
        parser.add_argument('--min_len', type=int, default=32)
        parser.add_argument('--max_new_tokens', type=int, default=128)
        parser.add_argument('--min_new_tokens', type=int, default=32)
        parser.add_argument('--length_penalty', type=float, default=0.8)
        parser.add_argument('--repetition_penalty', type=float, default=1.2)
        parser.add_argument('--llm_tune', type=str, default='lora')
        parser.add_argument('--peft_config', type=str, default='')
        parser.add_argument('--peft_dir', type=str, default='')

        parser.add_argument('--save_every_n_epochs', type=int, default=5)
        ## quantization
        parser.add_argument('--load_in_8bit', action='store_true', default=False)

        ## lora config
        parser.add_argument('--lora_r', type=int, default=8)
        parser.add_argument('--lora_alpha', type=int, default=32)
        parser.add_argument('--lora_dropout', type=int, default=0.1)

        # optimization
        parser.add_argument('--weight_decay', type=float, default=0.05, help='optimizer weight decay')
        parser.add_argument('--init_lr', type=float, default=1e-4, help='optimizer init learning rate')
        parser.add_argument('--min_lr', type=float, default=5e-6, help='optimizer min learning rate')
        parser.add_argument('--warmup_lr', type=float, default=1e-6, help='optimizer warmup learning rate')
        parser.add_argument('--warmup_steps', type=int, default=1000, help='optimizer warmup steps')
        parser.add_argument('--lr_decay_rate', type=float, default=0.9, help='optimizer lr decay rate')
        parser.add_argument('--scheduler', type=str, default='linear_warmup_cosine_lr', help='type of scheduler')
        parser.add_argument('--stage1_path', type=str, default='')
        parser.add_argument('--stage2_path', type=str, default='')
        parser.add_argument('--init_checkpoint', type=str, default='')
        parser.add_argument('--caption_eval_epoch', type=int, default=10)
        return parent_parser


