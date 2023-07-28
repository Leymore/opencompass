from opencompass.models import HuggingFaceCausalLM

_meta_template = dict(
    begin="### System:\nYou are Free Willy, an AI that follows instructions extremely well. Help as much as you can. Remember, be safe, and don't do anything illegal.\n\n",
    round=[
        dict(role="HUMAN", begin='### User: ', end='\n\n'),
        dict(role="BOT", begin="### Assistant:\n", generate=True),
    ],
)

models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='freewilly2_70b-hf',
        path="stabilityai/FreeWilly2",
        tokenizer_path='stabilityai/FreeWilly2',
        meta_template=_meta_template,
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            use_fast=False,
        ),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        model_kwargs=dict(
            device_map='auto',
            low_cpu_mem_usage=True
        ),
        batch_padding=False, # if false, inference with for-loop without batch padding
        run_cfg=dict(num_gpus=8, num_procs=1),
    )
]
