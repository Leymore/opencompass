import getpass
import json
import os
import os.path as osp
import uuid
import warnings
from datetime import datetime
from functools import partial
from typing import Dict, Union
import time

import mmengine
import requests
from mmengine.config import Config, ConfigDict

from opencompass.utils import build_dataset_from_cfg, dataset_abbr_from_cfg, model_abbr_from_cfg
from opencompass.openicl.icl_inferencer import PPLInferencer, GenInferencer
from mmengine.utils import track_parallel_progress


from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import base64

METRIC_WHITE_LIST = ['accuracy', 'rouge1', 'rouge2', 'rougeL', 'rougeLsum', 'score', 'humaneval_pass@1', 'auc_score', 'avg_toxicity_score', 'bleurt_diff', 'matthews_correlation', 'truth', 'info']
# WARNING!!!
# DO NOT CHANGE THIS KEY!!!
# 由于面向实验室内部多个团队使用，为了将来取数据的方便，这个 key 需统一使用且不能改变
INTERNAL_USAGE_KEY = b"InternalUsageKey"
index_url = "http://106.14.134.80:10824"

def decrypt_number(encrypted, key=INTERNAL_USAGE_KEY):
    raw = base64.b64decode(encrypted)
    cipher = AES.new(key, AES.MODE_ECB)
    decrypted = unpad(cipher.decrypt(raw), AES.block_size)
    return float(decrypted.decode())

def parse_score_dict(scores):
    ret = {}
    for k, v in scores.items():
        if v:
            ret[k] = decrypt_number(v)
    return ret

model_names = [
    'LLaMA-2-7B',
    'LLaMA-2-13B',
    'LLaMA-2-70B',
    # 'LLaMA-2-7B-Chat',
    # 'LLaMA-2-13B-Chat',
    # 'LLaMA-2-70B-Chat',
]

class DatabaseReporter:

    def __init__(self,
                 url: str,
                 retry: int = 2):
        self.url = url  # TODO: Add this to secret
        self.eval_toolkit_version = '0.5.0'  # TODO: flexible
        self.eval_person = getpass.getuser()  # Get the username of the current user
        self.retry = retry

    def post(self, endpoint: str, content: Union[str, Dict]):
        """Post a message to endpoint."""
        # return True
        if isinstance(content, dict):
            content = json.dumps(content)
        header = {'Content-Type': 'application/json'}
        url = osp.join(self.url, endpoint)
        retry = self.retry
        while True:
            try:
                r = requests.post(url, data=content, headers=header)
                break
            except Exception as e:
                retry -= 1
                if retry <= 0:
                    warnings.warn(
                        f'Error when posting to {url}: {e}\n'
                        f'Content: {content}')
                    return False, e
                else:
                    time.sleep(1)
        if r.status_code != 200:
            warnings.warn(
                f'Error when posting to {url}: {r.content.decode("utf-8")}\n'
                f'Content: {content}')

        return True, r.json()

    def run(self):
        for model_name in model_names:
            content = {
                "model_name": model_name,
                "model_version": "1.0.0",
            }
            success, model_ids = self.post("GetModelByNameAndVer", content)
            for model_id in model_ids:
                content = {
                    "model_id": model_id,
                }
                success, eval_task_ids = self.post("GetTaskListByModel", content)

                for eval_task_id in eval_task_ids:
                    if eval_task_id in ['284d146e-1b5d-5d31-b441-817a1228aa5c', '7e21ebd8-da73-5f95-a47e-1f3bd1f608f9']:
                        continue
                    content = {
                        "eval_task_id": eval_task_id
                    }
                    success, records = self.post("GetScoresByTask", content)
                    print(model_name, eval_task_id, len(records))
                    for record in records:
                        if record['model_id'] != model_id:
                            continue
                        print(record)
                        dataset_id = record['dataset_id']
                        content = {
                            "dataset_id": dataset_id
                        }
                        success, data_info = self.post("GetDatasetInfo", content)
                        assert success
                        dataset_name = data_info['name']
                        score = parse_score_dict(record['score'])

                        print(model_name, dataset_name, score)



DatabaseReporter(url=index_url).run()
