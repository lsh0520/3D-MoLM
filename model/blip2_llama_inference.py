import logging
import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType, PeftModel
from model.blip2 import Blip2Base
from transformers import AutoTokenizer
from transformers import LlamaForCausalLM
import argparse

class Blip2Llama(Blip2Base):
    def __init__(
        self,
        args=None,
    ):
        super().__init__()

        llm_model = args.llm_model
        self.graph_encoder, self.ln_graph, self.dictionary = self.init_unimol_encoder(args)
        
        self.Qformer, self.query_tokens = self.init_Qformer(args.bert_name, args.num_query_token, self.graph_encoder.num_features, args.cross_attention_freq)
        
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

        self.llm_model = LlamaForCausalLM.from_pretrained(llm_model, torch_dtype=torch.bfloat16)
        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))
        
        self.peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                                      inference_mode=False,
                                      r=args.lora_r,
                                      lora_alpha=args.lora_alpha,
                                      lora_dropout=args.lora_dropout,
                                      target_modules=['k_proj', 'v_proj', 'q_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])
        self.llm_model = get_peft_model(self.llm_model, self.peft_config)
        
        self.eos_token_id = self.llm_tokenizer.eos_token_id
        self.pad_token_id = self.llm_tokenizer.pad_token_id
        self.llm_tokenizer.padding_side = 'right'

        self.llm_proj = nn.Linear(self.Qformer.config.hidden_size, self.llm_model.config.hidden_size)

        if args.lora_path:
            self.load_ckpt(args.lora_path)

    def load_ckpt(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state_dict = ckpt['state_dict']
        state_dict = {k.split('blip2opt.')[1]: v for k, v in state_dict.items()}
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        if len(unexpected_keys) == 0:
            print(f"loaded model from {ckpt_path}")
        else:
            print(f"missing_keys: {missing_keys}, unexpected_keys: {unexpected_keys}")

    @torch.no_grad()
    def generate(
        self,
        graph_batch,
        text_batch,
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
        graph_embeds, graph_masks = self.graph_encoder(*graph_batch)
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



