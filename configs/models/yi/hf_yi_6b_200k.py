from opencompass.models import HuggingFace


_meta_template = dict(
    round=[
        dict(role="HUMAN", end='\n\n'),
        dict(role="BOT", begin="### Response:", end='</s>', generate=True),
    ],
)

models = [
    dict(
        type=HuggingFace,
        abbr='yi-6b-200k-hf',
        path='01-ai/Yi-6B-200K',
        tokenizer_path='01-ai/Yi-6B-200K',
        model_kwargs=dict(
            trust_remote_code=True,
            device_map='auto',
        ),
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
        ),
        meta_template=_meta_template,
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        run_cfg=dict(num_gpus=1, num_procs=1),
        end_str='</s>',
    )
]
