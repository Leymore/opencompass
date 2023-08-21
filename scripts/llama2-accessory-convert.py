import torch
from collections import OrderedDict
import os

output_basedir = "/mnt/petrelfs/zhoufengzhe/model_weights/llama2-accessory/sg/"
input_paths = [
    # "/mnt/petrelfs/share_data/hanjiaming/llama2_accessory_output/finetune/sg/dialog_sharegpt_epoch3_bs4_acc4_lr2e-5_minlr5e-6_warmp0.5_wd0/epoch2/consolidated.00-of-01.model.pth",
    # "/mnt/petrelfs/share_data/hanjiaming/llama2_accessory_output/finetune/sg/dialog_sharegpt_epoch4_bs16_acc16_lr1e-5_minlr5e-6_warmp0.5/epoch3/consolidated.00-of-01.model.pth",
    # "/mnt/petrelfs/share_data/hanjiaming/llama2_accessory_output/finetune/sg/dialog_sharegpt_epoch4_bs4_acc8_lr3e-5_minlr5e-6_warmp0.5/epoch3/consolidated.00-of-01.model.pth",
    # '/mnt/petrelfs/share_data/hanjiaming/llama2_accessory_output/finetune/sg/dialog_alpaca_epoch3_bs4_acc4_lr2e-5_minlr0_warmp0.04_wd0/epoch2/consolidated.00-of-01.model.pth',
    # '/mnt/petrelfs/share_data/hanjiaming/llama2_accessory_output/finetune/sg/dialog_sharegpt_epoch3_bs4_acc4_lr2e-5_minlr0_warmp0.04_wd0_tf32/epoch2/consolidated.00-of-01.model.pth',
    # '/mnt/petrelfs/share_data/hanjiaming/llama2_accessory_output/finetune/sg/dialog_sharegpt_epoch1_bs4_acc4_lr2e-5_minlr0_warmp0.04_wd0/epoch0/consolidated.00-of-01.model.pth',
    '/mnt/petrelfs/share_data/fga/eval/llama2-7B-dialog_4kflan/consolidated.00-of-01.model.pth',
    '/mnt/petrelfs/share_data/fga/eval/llama2-7B-dialog_4kmoss/consolidated.00-of-01.model.pth',
    '/mnt/petrelfs/share_data/fga/eval/llama2-7B-dialog_4kultra/consolidated.00-of-01.model.pth',
    '/mnt/petrelfs/share_data/fga/eval/llama2-7B-dialog_lima/consolidated.00-of-01.model.pth',
    '/mnt/petrelfs/share_data/fga/eval/llama2-7B-dialog_wizardcode/consolidated.00-of-01.model.pth',
    '/mnt/petrelfs/share_data/fga/eval/llama2-7B-dialog_wizardcode_loadcode220k/consolidated.00-of-01.model.pth',
    '/mnt/petrelfs/share_data/fga/eval/llama2-7B-dialog_wizardLM/consolidated.00-of-01.model.pth',
]
llama_7b_path = "/mnt/petrelfs/share_data/llm_llama/llama2_raw/llama-2-7b/consolidated.00.pth"

for input_path in input_paths:
    print("Loading checkpoint from {}".format(llama_7b_path))
    checkpoint_llama = torch.load(llama_7b_path, map_location='cpu')

    print("Loading checkpoint from {}".format(input_path))
    checkpoint = torch.load(input_path, map_location='cpu')
    checkpoint_type = None

    for k in checkpoint_llama:
        if 'llma.' + k in checkpoint['model']:
            # show
            checkpoint_avg =(checkpoint['model']['llma.' + k].float() ** 2).sum().item()
            checkpoint_llama_avg = (checkpoint_llama[k].float() ** 2).sum().item()
            if checkpoint_type is None:
                if checkpoint_avg > 0.5 * checkpoint_llama_avg:
                    checkpoint_type = 'llama'
                else:
                    checkpoint_type = 'diff'

            print(k, checkpoint_type, checkpoint_avg, checkpoint_llama_avg)

            # modify
            if checkpoint_type == 'llama':
                checkpoint_llama[k] = checkpoint['model']['llma.' + k]
            else:
                checkpoint_llama[k] = checkpoint_llama[k] + checkpoint['model']['llma.' + k]

    expname, epoch = input_path.split('/')[-3:-1]
    if epoch.startswith('epoch'):
        dirname = os.path.join(output_basedir, expname + '--' + epoch)
    else:
        dirname = os.path.join(output_basedir, epoch)

    output_path = os.path.join(dirname, 'consolidated.00.pth')
    print("Saving checkpoint to {}".format(output_path))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(checkpoint_llama, output_path)

    llama_7b_params_path = os.path.join(os.path.dirname(llama_7b_path), 'params.json')
    output_path = os.path.join(dirname, 'params.json')
    print("Saving params to {}".format(output_path))
    os.system('cp {} {}'.format(llama_7b_params_path, output_path))
