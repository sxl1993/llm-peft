#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import warnings
import torch
from typing import Optional
from peft import PeftType
from peft.config import PeftConfig 
from peft.peft_model import PeftModel, PeftModelForCausalLM
from peft.utils.other import _get_batch_size
from peft import LoraConfig, PrefixTuningConfig, TaskType, get_peft_model

class PeftModelForChatGLM(PeftModelForCausalLM):
    def __init__(self, model, peft_config: PeftConfig, adapter_name="default"):
        super().__init__(model, peft_config, adapter_name)

    def get_prompt(self, batch_size: int, task_ids: Optional[torch.Tensor] = None):
        """
        Returns the virtual prompts to use for Peft. Only applicable when `peft_config.peft_type != PeftType.LORA`.
        """
        peft_config = self.active_peft_config
        prompt_encoder = self.prompt_encoder[self.active_adapter]
        prompt_tokens = (
            self.prompt_tokens[self.active_adapter]
            .unsqueeze(0)
            .expand(batch_size, -1)
            .to(prompt_encoder.embedding.weight.device)
        )

        if peft_config.peft_type == PeftType.PREFIX_TUNING:
            prompt_tokens = prompt_tokens[:, : peft_config.num_virtual_tokens]
            if peft_config.inference_mode:
                past_key_values = prompt_encoder.embedding.weight.repeat(batch_size, 1, 1)
            else:
                past_key_values = prompt_encoder(prompt_tokens)
            if self.base_model_torch_dtype is not None:
                past_key_values = past_key_values.to(self.base_model_torch_dtype)
            past_key_values = past_key_values.view(
                batch_size,                                        
                peft_config.num_virtual_tokens,
                peft_config.num_layers * 2,
                peft_config.num_attention_heads,
                peft_config.token_dim // peft_config.num_attention_heads,
            )
            if peft_config.num_transformer_submodules == 2:
                past_key_values = torch.cat([past_key_values, past_key_values], dim=2)
            
            # TODO 适配chatglm2-6b
            past_key_values = past_key_values.permute([2, 1, 0, 3, 4]).split(
                peft_config.num_transformer_submodules * 2
            )
            return past_key_values
        else:
            if peft_config.peft_type == PeftType.MULTITASK_PROMPT_TUNING:
                prompts = prompt_encoder(prompt_tokens, task_ids)
            else:
                if peft_config.inference_mode:
                    prompts = prompt_encoder.embedding.weight.repeat(batch_size, 1, 1)
                else:
                    prompts = prompt_encoder(prompt_tokens)
            return prompts

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        task_ids=None,
        **kwargs,
    ):
        peft_config = self.active_peft_config
        if not peft_config.is_prompt_learning:
            if self.base_model.config.model_type == "mpt":
                if inputs_embeds is not None:
                    raise AssertionError("forward in MPTForCausalLM does not support inputs_embeds")
                return self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    **kwargs,
                )

            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )

        batch_size = _get_batch_size(input_ids, inputs_embeds)
  
        if kwargs.get("position_ids", None) is not None:
            warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
            kwargs["position_ids"] = None
        if kwargs.get("token_type_ids", None) is not None:
            warnings.warn("Token type ids are not supported for parameter efficient tuning. Ignoring token type ids")
            kwargs["token_type_ids"] = None
        kwargs.update(
            {
                "attention_mask": attention_mask,
                "labels": labels,
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
            }
        )

        if peft_config.peft_type == PeftType.PREFIX_TUNING:
            past_key_values = self.get_prompt(batch_size)
            return self.base_model(
                input_ids=input_ids, inputs_embeds=inputs_embeds, past_key_values=past_key_values, **kwargs
            )
        else:
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)
            # concat prompt labels
            if labels is not None:
                prefix_labels = torch.full((batch_size, peft_config.num_virtual_tokens), -100).to(labels.device)
                kwargs["labels"] = torch.cat((prefix_labels, labels), dim=1)
            prompts = self.get_prompt(batch_size=batch_size, task_ids=task_ids)
            prompts = prompts.to(inputs_embeds.dtype)
            inputs_embeds = torch.cat((prompts, inputs_embeds), dim=1)
            return self.base_model(inputs_embeds=inputs_embeds, **kwargs)
        
    def prepare_inputs_for_generation(self, *args, task_ids: torch.Tensor = None, **kwargs):
        peft_config = self.active_peft_config
        model_kwargs = self.base_model_prepare_inputs_for_generation(*args, **kwargs)
        if peft_config.is_prompt_learning:
            # if model_kwargs.get("attention_mask", None) is not None:
            #     prefix_attention_mask = torch.ones(
            #         model_kwargs["input_ids"].shape[0], peft_config.num_virtual_tokens
            #     ).to(model_kwargs["input_ids"].device)
            #     model_kwargs["attention_mask"] = torch.cat(
            #         (prefix_attention_mask, model_kwargs["attention_mask"]), dim=1
            #     )

            if model_kwargs.get("position_ids", None) is not None:
                warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
                model_kwargs["position_ids"] = None

            if kwargs.get("token_type_ids", None) is not None:
                warnings.warn(
                    "Token type ids are not supported for parameter efficient tuning. Ignoring token type ids"
                )
                kwargs["token_type_ids"] = None

            if model_kwargs["past_key_values"] is None and peft_config.peft_type == PeftType.PREFIX_TUNING:
                past_key_values = self.get_prompt(batch_size=model_kwargs["input_ids"].shape[0])
                model_kwargs["past_key_values"] = past_key_values
            else:
                if model_kwargs["past_key_values"] is None:
                    inputs_embeds = self.word_embeddings(model_kwargs["input_ids"])
                    prompts = self.get_prompt(batch_size=model_kwargs["input_ids"].shape[0], task_ids=task_ids)
                    prompts = prompts.to(inputs_embeds.dtype)
                    model_kwargs["inputs_embeds"] = torch.cat((prompts, inputs_embeds), dim=1)
                    model_kwargs["input_ids"] = None

        return model_kwargs
        

def get_prefix_tuning2_model(model, model_name, pre_seq_len, token_dim, num_attention_heads):
    config = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM,
                                num_virtual_tokens=pre_seq_len,
                                token_dim=token_dim,
                                num_attention_heads=num_attention_heads,
                                inference_mode=False)
    if model_name not in ["chatglm2-6b", "chatglm3-6b"]:
        model = get_peft_model(model, config)    
    return model
    
def get_lora_model(model, model_name, target_modules, lora_rank, lora_dropout):
    target_modules = target_modules.split(",")
    config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                        lora_alpha=2 * lora_rank,
                        target_modules=target_modules,
                        inference_mode=False,
                        r=lora_rank,
                        lora_dropout=lora_dropout)
    model = get_peft_model(model, config)
    return model


if __name__ == "__main__":
    modules = "query_key_value"
    
    print(modules.split(","))
    
