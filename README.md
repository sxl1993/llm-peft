# 下载数据集
[AdvertiseGen](https://cloud.tsinghua.edu.cn/seafhttp/files/70242f6d-4287-4d44-813d-1188df86d285/AdvertiseGen.tar.gz)

# ChatGLM2-6b ptuning-v2

CUDA_VISIBLE_DEVICES=1 python3 finetune_generation.py ./chatglm2/pt_argument.json

# ChatGLM2-6b lora
CUDA_VISIBLE_DEVICES=1 python3 finetune_generation.py ./chatglm2/lora_argument.json

# CHatGLM3-6b ptuning-v2

CUDA_VISIBLE_DEVICES=1 python3 finetune_generation.py ./chatglm3/pt_argument.json

# ChatGLM3-6b lora
CUDA_VISIBLE_DEVICES=1 python3 finetune_generation.py ./chatglm3/lora_argument.json

# baichuan2-7b lora
CUDA_VISIBLE_DEVICES=1 python3 finetune_generation.py ./baichuan2/lora_argument.json