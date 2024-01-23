"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType, PeftModel

# from lavis.models.blip2_models.blip2 import (
#     # Blip2Base,
#     disabled_train,
# )
from model.blip2 import Blip2Base
# from transformers import LlamaTokenizer
from transformers import AutoTokenizer
# from model.modeling_llama import LlamaForCausalLM
from transformers import LlamaForCausalLM


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class Blip2Llama(Blip2Base):
    """
    BLIP2 first-stage model with Q-former and ViT.
    Supported model types:
        - pretrained: pretrained model with vit-g
        - pretrain_vitL: pretrained model with vit-large
        - coco: fintuned model on coco
    """
    def __init__(
        self,
        bert_name,
        gin_num_layers,
        gin_hidden_dim,
        gin_drop_ratio,
        tune_gnn=False,
        num_query_token=32,
        cross_attention_freq=2,
        llm_tuning='full',
        llm_model="decapoda-research/llama-7b-hf",
        args=None,
    ):
        super().__init__()

        self.args = args
        if args.use_3d:
            self.graph_encoder, self.ln_graph, self.dictionary = self.init_unimol_encoder(args)
        else:
            self.graph_encoder, self.ln_graph = self.init_graph_encoder(gin_num_layers, gin_hidden_dim, gin_drop_ratio)
        self.tune_gnn = tune_gnn
        if not tune_gnn:
            for name, param in self.graph_encoder.named_parameters():
                param.requires_grad = False
            self.graph_encoder = self.graph_encoder.eval()
            self.graph_encoder.train = disabled_train
            logging.info("freeze graph encoder")
        
        self.Qformer, self.query_tokens = self.init_Qformer(bert_name, num_query_token, self.graph_encoder.num_features, cross_attention_freq)
        ### remove the unused parameters
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        ## initialize opt model
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model, use_fast=False, padding_side='right')
        self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llm_tokenizer.add_special_tokens({'bos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'eos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'unk_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'additional_special_tokens': ['<mol>']})
        self.llm_tokenizer.mol_token_id = self.llm_tokenizer("<mol>", add_special_tokens=False).input_ids[0]

        if args.enable_flash:
            self.llm_model = LlamaForCausalLM.from_pretrained(llm_model, torch_dtype=torch.bfloat16)
        else:
            self.llm_model = LlamaForCausalLM.from_pretrained(llm_model, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))
        
        self.llm_tuning = llm_tuning
        if llm_tuning == 'lora':
            if args.peft_dir:
                self.llm_model = PeftModel.from_pretrained(self.llm_model, args.peft_dir, is_trainable=True)
            else:
                if self.args.peft_config:
                    peft_config = LoraConfig(**LoraConfig.from_json_file(self.args.peft_config))
                else:
                    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                                             inference_mode=False,
                                             r=args.lora_r,
                                             lora_alpha=args.lora_alpha,
                                             lora_dropout=args.lora_dropout,
                                             target_modules=['k_proj', 'v_proj', 'q_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])
                self.peft_config = peft_config
                self.llm_model = get_peft_model(self.llm_model, peft_config)
            self.llm_model.print_trainable_parameters()
        elif llm_tuning == 'full':
            pass
        elif llm_tuning == 'freeze':
            for name, param in self.llm_model.named_parameters():
                param.requires_grad = False
        else:
            raise NotImplementedError()
        
        self.eos_token_id = self.llm_tokenizer.eos_token_id
        self.pad_token_id = self.llm_tokenizer.pad_token_id

        self.llm_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llm_model.config.hidden_size
        )

    def forward(self, batch):
        graph_batch, text_batch = batch
        if self.args.use_3d:
            graph_embeds, graph_masks = self.graph_encoder(*graph_batch)
        else:
            graph_embeds, graph_masks = self.graph_encoder(graph_batch)
        if not self.tune_gnn:
            graph_embeds = graph_embeds.detach()
        graph_embeds = self.ln_graph(graph_embeds)
        query_tokens = self.query_tokens.expand(graph_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=graph_embeds,
            encoder_attention_mask=graph_masks,
            return_dict=True,
        )
        query_output = self.llm_proj(query_output.last_hidden_state) #[batch_size,num_query_token,dim]

        targets = text_batch.input_ids.masked_fill(
            text_batch.input_ids == self.llm_tokenizer.pad_token_id, -100
        ) # [batch_size, max_len]
        targets = targets.masked_fill(text_batch.token_type_ids == 0, -100)

        inputs_embeds = self.llm_model.get_input_embeddings()(text_batch.input_ids) # [batch_size, max_len, dim]
        inputs_embeds[text_batch.is_mol_token] = query_output.flatten(0, 1) # [batch_size, max_len, dim]


        outputs = self.llm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=text_batch.attention_mask,
            return_dict=True,
            labels=targets,
            use_cache=False,
        )
        loss = outputs.loss
        return {"loss": loss}

    @torch.no_grad()
    def generate(
        self,
        samples,
        do_sample=False,
        num_beams=5,
        max_length=128,
        min_length=1,
        max_new_tokens=128,
        min_new_tokens=32,
        repetition_penalty=1.2,
        length_penalty=1.0,
        num_captions=1,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        graph_batch = samples['graphs']
        text_batch = samples['prompt_tokens']
        
        # with self.maybe_autocast():
        if self.args.use_3d:
            graph_embeds, graph_masks = self.graph_encoder(*graph_batch)
        else:
            graph_embeds, graph_masks = self.graph_encoder(graph_batch)
        graph_embeds = self.ln_graph(graph_embeds)
        query_tokens = self.query_tokens.expand(graph_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=graph_embeds,
            encoder_attention_mask=graph_masks,
            return_dict=True,
        )
        query_output = self.llm_proj(query_output.last_hidden_state) #[batch_size,num_query_token,dim]
        inputs_embeds = self.llm_model.get_input_embeddings()(text_batch.input_ids) # [batch_size, max_len, dim]
        inputs_embeds[text_batch.is_mol_token] = query_output.flatten(0, 1) # [batch_size, max_len, dim]

        outputs = self.llm_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=text_batch.attention_mask,
            do_sample=do_sample,
            num_beams=num_beams,
            max_length=max_length,
            # min_length=min_length,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_captions,
        )
        output_text = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)

        output_text = [text.strip() for text in output_text]
        return output_text

