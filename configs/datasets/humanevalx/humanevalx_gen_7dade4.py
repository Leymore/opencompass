from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import HumanevalXDataset, HumanevalXEvaluator, humaneval_postprocess_v2


# Please download the needed `xx.jsonl.gz` from
# https://github.com/THUDM/CodeGeeX2/tree/main/benchmark/humanevalx
# and move them into `data/humanevalx/` folder

humanevalx_datasets = []
for lang, lang_str in [
    ['python', 'Python'],
    ['cpp', 'C++'],
    ['go', 'Go'],
    ['java', 'Java'],
    ['js', 'JavaScript'],
]:

    humanevalx_reader_cfg = dict(input_columns=['prompt'], output_column='task_id', train_split='test')

    humanevalx_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                round=[
                    dict(role='HUMAN', prompt=f'You are an intelligent programming assistant to produce {lang_str} algorithmic solutions.\nCan you complete the following {lang_str} function?\n```{lang}\n{{prompt}}\n```'),
                ]
            )),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer, max_out_len=1024),
    )

    humanevalx_eval_cfg = dict(
        evaluator=dict(
            type=HumanevalXEvaluator,
            language=lang,
            # EXTERNAL ip_address='localhost',
            ip_address='http://service.opencompass.org.cn',  # INTERNAL
            port=5001,
        ),  # refer to https://opencompass.readthedocs.io/en/latest/advanced_guides/code_eval_service.html to launch a server
        pred_role='BOT',
        pred_postprocessor=dict(type=humaneval_postprocess_v2),
    )

    humanevalx_datasets.append(
        dict(
            type=HumanevalXDataset,
            abbr=f'humanevalx-{lang}',
            language=lang,
            path='./data/humanevalx',
            reader_cfg=humanevalx_reader_cfg,
            infer_cfg=humanevalx_infer_cfg,
            eval_cfg=humanevalx_eval_cfg
        )
    )
