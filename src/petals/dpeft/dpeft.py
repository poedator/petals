import torch
import inspect

from transformers import PreTrainedModel
from transformers.utils import PushToHubMixin

from peft import (PeftConfig, PeftModel, PeftModelForCausalLM, PeftModelForSequenceClassification,
                  PromptTuningConfig, PrefixTuningConfig)
from peft.utils.config import PeftType, TaskType, PromptLearningConfig
from peft.tuners.prompt_tuning import PromptEmbedding
from peft.utils.other import _prepare_prompt_learning_config

from typing import Any, Dict, List, Optional, Union



class DistributedPeftModel(PeftModel):
    def __init__(self, model: PreTrainedModel, peft_config: PeftConfig, adapter_name: str = "default"):
        
        assert getattr(model.config, 'tuning_mode', None) is None, "this model is already pre-configured for tuning. Set `tuning_mode` to None"
        
        if isinstance(peft_config, PromptTuningConfig):
            super().__init__(model, peft_config, adapter_name)
        elif isinstance(peft_config, PrefixTuningConfig):
            super().__init__(model, peft_config, adapter_name)
        else:
            raise NotImplementedError("Only prompt tuning and prefix tuning are supported for now")


    def _prefix_tuning_forward(self, *args, **kwargs):

        # prefixes will be passed in form of past_key_values argument
        # this is supported by Bloom and Llama. 
        # if not supported - see workaround in the original peft code
        # https://github.com/huggingface/peft/blob/ec267c644a9a9f05a7340a7cb23ed5a6a6090dd0/src/peft/peft_model.py#L834
        fwd_params = list(inspect.signature(self.base_model.forward).parameters.keys())
        assert "past_key_values" in fwd_params, "Model not supported, past_key_values not in forward() params"

        return super().prefix_tuning_forward(*args, **kwargs)


class DistributedPeftModelForCausalLM(PeftModelForCausalLM, DistributedPeftModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class DistributedPeftModelForSequenceClassification(PeftModelForSequenceClassification, DistributedPeftModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


MODEL_TYPE_TO_PEFT_MODEL_MAPPING = {
    "SEQ_CLS": DistributedPeftModelForSequenceClassification,
    "CAUSAL_LM": DistributedPeftModelForCausalLM,
    # "SEQ_2_SEQ_LM": DistributedPeftModelForSeq2SeqLM,
    # "TOKEN_CLS": DistributedPeftModelForTokenClassification,
    # "QUESTION_ANS": DistributedPeftModelForQuestionAnswering,
    # "FEATURE_EXTRACTION": DistributedPeftModelForFeatureExtraction,
}


def get_peft_model(model: PreTrainedModel, peft_config: PeftConfig, adapter_name: str = "default") -> DistributedPeftModel:
    # NOTE: from peft/mapping.py
    """
    Returns a DistributedPeftModel model object from a Petals Distributed model and a PeftConfig.

    Args:
        model ([`transformers.PreTrainedModel`]): Model to be wrapped.
        peft_config ([`PeftConfig`]): Configuration object containing the parameters of the Peft model.
    """

    model_config = getattr(model, "config", {"model_type": "custom"})
    if hasattr(model_config, "to_dict"):
        model_config = model_config.to_dict()

    peft_config.base_model_name_or_path = model.__dict__.get("name_or_path", None)

    if peft_config.task_type not in MODEL_TYPE_TO_PEFT_MODEL_MAPPING.keys() and not isinstance(
        peft_config, PromptLearningConfig
    ):
        raise NotImplementedError (f"Task type {peft_config.task_type} not supported")
        # return PeftModel(model, peft_config, adapter_name=adapter_name)  # ORIGNAL
    
    if isinstance(peft_config, PromptLearningConfig):
        pass
        # peft_config = _prepare_prompt_learning_config(peft_config, model_config)
        # adds token_dim, num_layers, num_attention_heads, encoder_hidden_size to peft_config
    else:
        raise NotImplementedError
        pass

    selected_constructor = MODEL_TYPE_TO_PEFT_MODEL_MAPPING[peft_config.task_type]
    print(f"{selected_constructor=}")
    return selected_constructor(model, peft_config, adapter_name=adapter_name)


