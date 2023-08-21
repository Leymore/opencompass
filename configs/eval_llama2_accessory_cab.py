from mmengine.config import read_base
from opencompass.models import Llama2

with read_base():
    from .datasets.XLSum.XLSum_gen_2bb71c import XLSum_datasets
    from .datasets.cmmlu.cmmlu_ppl_8b9c76 import cmmlu_datasets
    from .datasets.xiezhi.xiezhi_ppl_ea6bd7 import xiezhi_datasets
    from .datasets.tydiqa.tydiqa_gen_978d2a import tydiqa_datasets
    from .datasets.squad20.squad20_gen_1710bc import squad20_datasets
    from .datasets.drop.drop_gen_599f07 import drop_datasets
    from .datasets.XCOPA.XCOPA_ppl_54058d import XCOPA_datasets
    from .datasets.collections.base_medium_llama import datasets

    from .summarizers.medium import summarizer


datasets = sum([v for k, v in locals().items() if k.endswith("_datasets") or k == 'datasets'], [])
datasets = [i for i in datasets if i.get('abbr', '').startswith('ceval-')][:1]
# datasets = [i for i in datasets if i.get('abbr', '').startswith('ceval-')]
# datasets = [i for i in datasets if i.get('abbr', '').startswith('lukaemon_mmlu')][:10]
# datasets = [i for i in datasets if not i.get('abbr', '').startswith('GaokaoBench')]

for d in datasets:
    d["infer_cfg"]["inferencer"]["save_every"] = 1

# _meta_template = dict(
#     begin="A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n###",
#     round=[
#         dict(role='HUMAN', begin='Human: ', end='\n###'),
#         dict(role='BOT', begin='Assistant:', end='\n###', generate=True),
#     ]
# )

names = [
    'dialog_sharegpt_epoch3_bs4_acc4_lr2e-5_minlr5e-6_warmp0.5_wd0--epoch2',
    'dialog_sharegpt_epoch4_bs16_acc16_lr1e-5_minlr5e-6_warmp0.5--epoch3',
    'dialog_sharegpt_epoch4_bs4_acc8_lr3e-5_minlr5e-6_warmp0.5--epoch3',
    'dialog_alpaca_epoch3_bs4_acc4_lr2e-5_minlr0_warmp0.04_wd0--epoch2',
    'dialog_sharegpt_epoch3_bs4_acc4_lr2e-5_minlr0_warmp0.04_wd0_tf32--epoch2',
    'dialog_sharegpt_epoch1_bs4_acc4_lr2e-5_minlr0_warmp0.04_wd0--epoch0',
    'llama2-7B-dialog_4kflan',
    'llama2-7B-dialog_4kmoss',
    'llama2-7B-dialog_4kultra',
    'llama2-7B-dialog_lima',
    'llama2-7B-dialog_wizardcode',
    'llama2-7B-dialog_wizardcode_loadcode220k',
    'llama2-7B-dialog_wizardLM',
]

models = []
for name in names:
    model_cfg = dict(
        abbr=name,
        type=Llama2, path=f'/mnt/petrelfs/zhoufengzhe/model_weights/llama2-accessory/sg/{name}',
        tokenizer_path='/mnt/petrelfs/share_data/basemodel/checkpoints/llm/llama2/llama/tokenizer.model',
        # meta_template=_meta_template,
        max_out_len=100, max_seq_len=2048, batch_size=16, run_cfg=dict(num_gpus=1, num_procs=1))
    models.append(model_cfg)

work_dir = './outputs/debug/llama-2-accessory-cab/'
