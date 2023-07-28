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
# index_url = "http://106.14.134.80:10824"
index_url = "http://inferstore-eval.openxlab.org.cn"

def encrypt_number(number, key=INTERNAL_USAGE_KEY):
    cipher = AES.new(key, AES.MODE_ECB)
    encrypted = cipher.encrypt(pad(str(number).encode(), AES.block_size))
    return base64.b64encode(encrypted).decode()


class UploadedResultsCache:
    def __init__(self,
                 uploaded_result_path: str = '.cache/uploaded_results.json'):
        self.uploaded_result_path = uploaded_result_path

    @property
    def uploaded_results(self):
        if not hasattr(self, '_uploaded_results'):
            if osp.exists(self.uploaded_result_path):
                self._uploaded_results = mmengine.load(self.uploaded_result_path)
            else:
                self._uploaded_results = {
                    'model_ids': [],
                    'dataset_ids': [],
                    'record_ids': []
                }
        return self._uploaded_results

    @property
    def model_ids(self):
        return self.uploaded_results['model_ids']

    @property
    def dataset_ids(self):
        return self.uploaded_results['dataset_ids']

    @property
    def record_ids(self):
        return self.uploaded_results['record_ids']

    def save(self):
        mmengine.dump(self.uploaded_results,
                      self.uploaded_result_path,
                      indent=4,
                      ensure_ascii=False)


def get_uuid(s):
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, 'magic' + s))

class DatabaseReporter:

    def __init__(self,
                 url: str,
                 work_dir: str = '/mnt/petrelfs/zhoufengzhe/repos/pjeval/outputs/20230727/20230727_200652',
                 uploaded_result_path: str = '.cache/uploaded_results.json',
                 retry: int = 2):
        self.work_dir = work_dir
        self.url = url  # TODO: Add this to secret
        self.eval_toolkit_version = '0.5.0'  # TODO: flexible
        self.eval_person = getpass.getuser()  # Get the username of the current user
        self.dataset_size_path = '.cache/dataset_size.json'
        self.uploaded_results = UploadedResultsCache(uploaded_result_path)
        self.retry = retry

    def convert_to_timestamp(self,
                             date_str,
                             date_format='%Y%m%d_%H%M%S') -> str:
        dt = datetime.strptime(date_str, date_format)
        timestamp = dt.timestamp()
        return str(int(timestamp))

    @property
    def dataset_size(self):
        if not hasattr(self, '_dataset_size'):
            if osp.exists(self.dataset_size_path):
                self._dataset_size = mmengine.load(self.dataset_size_path)
            else:
                self._dataset_size = {}
        return self._dataset_size

    def create_model(self, model_cfg):
        model_name = model_abbr_from_cfg(model_cfg)
        model_version = "1.0.0"
        model_prompt = json.dumps(model_cfg.get('meta_template', {}), sort_keys=True)
        model_id = get_uuid(model_name + '--' + model_version + '--' + model_prompt)
        if model_id not in self.uploaded_results.model_ids:
            content = {
                'model_id': model_id,
                'model_name': model_name,
                'model_version': model_version,
                'model_prompt': model_prompt
            }
            out = self.post('CreateModel', content)
            if not out[0]:
                return out[1]
        return model_id

    def create_dataset(self, dataset_cfg):
        dataset_prompt = json.dumps(dataset_cfg.to_dict(), sort_keys=True)
        dataset_prompt_id = get_uuid(dataset_prompt)
        dataset_name = dataset_abbr_from_cfg(dataset_cfg)
        if 'PPLInferencer' in dataset_cfg['infer_cfg']['inferencer']['type']:
            dataset_type = 'discriminative'
        elif 'GenInferencer' in dataset_cfg['infer_cfg']['inferencer']['type']:
            dataset_type = 'generate'
        else:
            dataset_type = 'unknown'
        dataset_version = "1.0.0"
        dataset_item_cnt = self.dataset_size[dataset_name]
        dataset_tags = [dataset_name]
        dataset_id = get_uuid(dataset_name + '--' + dataset_type + '--' + dataset_version + '--' + str(dataset_item_cnt) + '--' + str(dataset_tags) + '--' + str(dataset_prompt))

        if dataset_id not in self.uploaded_results.dataset_ids:
            content = {
                "dataset_prompt_id": dataset_prompt_id,
                "dataset_prompt": dataset_prompt,
            }
            out = self.post("CreateDatasetPrompt", content)
            if not out[0]:
                return out[1]

            content = {
                "dataset_id": dataset_id,
                "dataset_name": dataset_name,
                "dataset_type": dataset_type,
                "dataset_version": dataset_version,
                "dataset_item_cnt": dataset_item_cnt,
                "dataset_tags": dataset_tags,
                "dataset_prompt": [dataset_prompt_id],
            }
            out = self.post("CreateEvalDataset", content)
            if not out[0]:
                return out[1]
        return dataset_id

    def create_obj_eval_task(self, date_str, model_id, model_cfg, dataset_id, dataset_cfg):
        model_abbr = model_abbr_from_cfg(model_cfg)
        dataset_abbr = dataset_abbr_from_cfg(dataset_cfg)
        result_file = osp.join(self.work_dir, 'results', model_abbr, dataset_abbr + ".json")
        if not osp.exists(result_file):
            return
        eval_toolkit_version = self.eval_toolkit_version
        eval_start_ts = self.convert_to_timestamp(date_str)
        eval_end_ts = str(int(datetime.now().timestamp()))
        eval_person = self.eval_person
        with open(result_file, 'r') as f:
            eval_scores = json.load(f)
        eval_scores = {k: encrypt_number(round(float(v), 4)) for k, v in eval_scores.items() if k in METRIC_WHITE_LIST}
        eval_task_id = get_uuid(eval_start_ts + '--' + eval_person + '--' + eval_toolkit_version + '--' + model_id)
        record_id = get_uuid(eval_task_id + '--' + dataset_id)
        if record_id not in self.uploaded_results.record_ids:
            content = {
                "eval_task_id": eval_task_id,
                "model_id": model_id,
                "dataset_id": dataset_id,
                "eval_toolkit_version": eval_toolkit_version,
                "eval_start_ts": eval_start_ts,
                "eval_end_ts": eval_end_ts,
                "eval_person": eval_person,
                "eval_scores": eval_scores,
            }
            out = self.post("CreateObjEvalTask", content)
            if not out[0]:
                return out[1]
        return record_id

    def _create_obj_eval_task(self, args):
        return self.create_obj_eval_task(*args)

    def run(self):
        # get lateset configs
        config_name = sorted(os.listdir(osp.join(self.work_dir, "configs")))[-1]
        cfg = Config.fromfile(osp.join(self.work_dir, "configs", config_name), format_python_code=False)
        timestamp = osp.basename(osp.normpath(self.work_dir))
        error = False

        model_ids = track_parallel_progress(self.create_model, cfg['models'], nproc=16, keep_order=False)
        for model_id in model_ids:
            if isinstance(model_id, str):
                if model_id not in self.uploaded_results.model_ids:
                    self.uploaded_results.model_ids.append(model_id)
            else:
                print(f'something wrong when creating model: {str(model_id)}')
                error = True
        self.uploaded_results.save()
        if error:
            return

        dataset_ids = track_parallel_progress(self.create_dataset, cfg['datasets'], nproc=16, keep_order=False)
        # dataset_ids = [self.create_dataset(i) for i in cfg['datasets']]
        for dataset_id in dataset_ids:
            if isinstance(dataset_id, str):
                if dataset_id not in self.uploaded_results.dataset_ids:
                    self.uploaded_results.dataset_ids.append(dataset_id)
            else:
                print(dataset_id)
                print(f'something wrong when creating dataset: {str(dataset_id)}')
        self.uploaded_results.save()
        if error:
            return

        tasks = []
        for model_id, model_cfg in zip(model_ids, cfg['models']):
            for dataset_id, dataset_cfg in zip(dataset_ids, cfg['datasets']):
                tasks.append((timestamp, model_id, model_cfg, dataset_id, dataset_cfg))
        record_ids = track_parallel_progress(self._create_obj_eval_task, tasks, nproc=16, keep_order=False)
        for record_id in record_ids:
            if isinstance(record_id, str):
                if record_id not in self.uploaded_results.record_ids:
                    self.uploaded_results.record_ids.append(record_id)
            elif record_id is None: # results doesn't exist
                continue
            else:
                print(f'something wrong when creating record: {str(record_id)}')
        self.uploaded_results.save()

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
                    return False, (content, e)
                else:
                    time.sleep(1)
        if r.status_code != 200:
            resp_content = r.content.decode("utf-8")
            if 'Duplicate' not in resp_content:
                warnings.warn(
                    f'Error when posting to {url}: {resp_content}\n'
                    f'Content: {content}')
                return False, Exception(resp_content)
        return True, (content, None)

DatabaseReporter(url=index_url).run()
