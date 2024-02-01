from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import TheoremQADataset, TheoremQA_postprocess

TheoremQA_reader_cfg = dict(
    input_columns=['Question', 'Answer_type'],
    output_column='Answer',
    train_split='test')

TheoremQA_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role='HUMAN',
                prompt=
                """You are a mathematician, you are supposed to answer the given question. You need to output the answer in your final sentence like "Therefore, the answer is ...". The answer can only be one of the following forms:\n1. a numerical value like 0.1, no symbol and no unit at all.\n2. a list of number like [2, 3, 4].\n3. True/False.\n4. an option like (a), (b), (c), (d)\nQuestion: {Question}\nLet\'s think step by step."""
            ),
        ])),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512))

TheoremQA_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_postprocessor=dict(type=TheoremQA_postprocess))

TheoremQA_datasets = [
    dict(
        abbr='TheoremQA',
        type=TheoremQADataset,
        path='./data/TheoremQA/test.csv',
        reader_cfg=TheoremQA_reader_cfg,
        infer_cfg=TheoremQA_infer_cfg,
        eval_cfg=TheoremQA_eval_cfg)
]
