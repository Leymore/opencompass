from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("model")
parser.add_argument("revision", nargs="?", default="main")
args = parser.parse_args()
model = AutoModelForCausalLM.from_pretrained(
    args.model,
    trust_remote_code=True,
    revision=args.revision,
    cache_dir="/mnt/petrelfs/share_data/basemodel/checkpoints/llm/hf_hub"
)
tokenizer = AutoTokenizer.from_pretrained(
    args.model,
    trust_remote_code=True,
    cache_dir="/mnt/petrelfs/share_data/basemodel/checkpoints/llm/hf_hub"
)

"""
srun -p llmeval --quotatype=auto --gres=gpu:0 -N1 python3 scripts/download_hf_models.py internlm/internlm-7b
srun -p llmeval --quotatype=auto --gres=gpu:0 -N1 python3 scripts/download_hf_models.py internlm/internlm-chat-7b
srun -p llmeval --quotatype=auto --gres=gpu:0 -N1 python3 scripts/download_hf_models.py internlm/internlm-chat-7b-8k
srun -p llmeval --quotatype=auto --gres=gpu:0 -N1 python3 scripts/download_hf_models.py huggyllama/llama-7b
srun -p llmeval --quotatype=auto --gres=gpu:0 -N1 python3 scripts/download_hf_models.py huggyllama/llama-13b
srun -p llmeval --quotatype=auto --gres=gpu:0 -N1 python3 scripts/download_hf_models.py huggyllama/llama-30b
srun -p llmeval --quotatype=auto --gres=gpu:0 -N1 python3 scripts/download_hf_models.py huggyllama/llama-65b
srun -p llmeval --quotatype=auto --gres=gpu:0 -N1 python3 scripts/download_hf_models.py meta-llama/Llama-2-7b-hf
srun -p llmeval --quotatype=auto --gres=gpu:0 -N1 python3 scripts/download_hf_models.py meta-llama/Llama-2-13b-hf
srun -p llmeval --quotatype=auto --gres=gpu:0 -N1 python3 scripts/download_hf_models.py meta-llama/Llama-2-70b-hf
srun -p llmeval --quotatype=auto --gres=gpu:0 -N1 python3 scripts/download_hf_models.py meta-llama/Llama-2-7b-chat-hf
srun -p llmeval --quotatype=auto --gres=gpu:0 -N1 python3 scripts/download_hf_models.py meta-llama/Llama-2-13b-chat-hf
srun -p llmeval --quotatype=auto --gres=gpu:0 -N1 python3 scripts/download_hf_models.py meta-llama/Llama-2-70b-chat-hf
srun -p llmeval --quotatype=auto --gres=gpu:0 -N1 python3 scripts/download_hf_models.py tiiuae/falcon-7b
srun -p llmeval --quotatype=auto --gres=gpu:0 -N1 python3 scripts/download_hf_models.py tiiuae/falcon-40b
srun -p llmeval --quotatype=auto --gres=gpu:0 -N1 python3 scripts/download_hf_models.py TigerResearch/tigerbot-7b-base
srun -p llmeval --quotatype=auto --gres=gpu:0 -N1 python3 scripts/download_hf_models.py TigerResearch/tigerbot-7b-sft
srun -p llmeval --quotatype=auto --gres=gpu:0 -N1 python3 scripts/download_hf_models.py mosaicml/mpt-7b
srun -p llmeval --quotatype=auto --gres=gpu:0 -N1 python3 scripts/download_hf_models.py mosaicml/mpt-7b-instruct
srun -p llmeval --quotatype=auto --gres=gpu:0 -N1 python3 scripts/download_hf_models.py fnlp/moss-moon-003-base
srun -p llmeval --quotatype=auto --gres=gpu:0 -N1 python3 scripts/download_hf_models.py fnlp/moss-moon-003-sft
srun -p llmeval --quotatype=auto --gres=gpu:0 -N1 python3 scripts/download_hf_models.py lmsys/vicuna-7b-v1.3
srun -p llmeval --quotatype=auto --gres=gpu:0 -N1 python3 scripts/download_hf_models.py lmsys/vicuna-13b-v1.3
srun -p llmeval --quotatype=auto --gres=gpu:0 -N1 python3 scripts/download_hf_models.py lmsys/vicuna-33b-v1.3
srun -p llmeval --quotatype=auto --gres=gpu:0 -N1 python3 scripts/download_hf_models.py TheBloke/wizardLM-7B-HF
srun -p llmeval --quotatype=auto --gres=gpu:0 -N1 python3 scripts/download_hf_models.py baichuan-inc/baichuan-7B
srun -p llmeval --quotatype=auto --gres=gpu:0 -N1 python3 scripts/download_hf_models.py baichuan-inc/Baichuan-13B-Base
srun -p llmeval --quotatype=auto --gres=gpu:0 -N1 python3 scripts/download_hf_models.py baichuan-inc/Baichuan-13B-Chat
srun -p llmeval --quotatype=auto --gres=gpu:0 -N1 python3 scripts/download_hf_models.py FlagAlpha/Llama2-Chinese-7b-Chat
srun -p llmeval --quotatype=auto --gres=gpu:0 -N1 python3 scripts/download_hf_models.py FlagAlpha/Llama2-Chinese-13b-Chat
srun -p llmeval --quotatype=auto --gres=gpu:0 -N1 python3 scripts/download_hf_models.py stabilityai/FreeWilly2
"""
