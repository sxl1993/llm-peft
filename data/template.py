#!/usr/bin/env python3
# coding=utf-8

from dataclasses import dataclass
from typing import List, Dict, Union
from transformers import PreTrainedTokenizer

@dataclass
class Template:
    prompt: List[Union[str, Dict[str, str]]]


templates: Dict[str, Template] = {}

def register_template(
        name: str, 
        prompt: List[Union[str, Dict[str, str]]]
    ):
    templates[name] = Template(prompt)


def get_template_and_fix_tokenizer(
        name: str,
        tokenizer: PreTrainedTokenizer
        ) -> Template:
    template = templates.get(name, None)
    assert template is not None, "Template {} does not exist.".format(name)

    return template


register_template(
    name="chatglm2",
    prompt=[
        "[Round {{idx}}]\n\n问：{{query}}\n\n答："
    ],
)

register_template(
    name="chatglm3",
    prompt=[
        {"token": "<|user|>"},
        "\n",
        "{{query}}",
        {"token": "<|assistant|>"},
        "\n" # add an extra newline to avoid error in ChatGLM's process_response method
    ],
)
