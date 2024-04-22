from opencompass.models import HuggingFaceAboveV433Base


models = [
    dict(
        type=HuggingFaceAboveV433Base,
        abbr='mistral-7b-v0.1-hf',
        path='mistralai/Mistral-7B-v0.1',
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=1),
    )
]
