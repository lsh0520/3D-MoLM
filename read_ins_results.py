from transformers import AutoTokenizer
import json
import os
import re
import numpy as np
from collections import Counter
from model.help_funcs import caption_evaluate

root = 'all_checkpoints/stage3_test/lightning_logs/version_3/'
file_path = os.path.join(root, 'predictions.txt')

dicts = []
with open(file_path, 'r', encoding='utf8') as f:
    for line in f:
        dicts.append(json.loads(line.strip()))

_prediction = []
_targets = []
_task_type = []
for _dict in dicts:
    _prediction.extend(_dict['prediction'])
    _targets.extend(_dict['target'])
    _task_type.extend(_dict['task_type'])

task_type_counter = Counter(_task_type)
prediction = {}
targets = {}

prediction['HOMO'] = []
prediction['LUMO'] = []
prediction['HOMO-LUMO Gap'] = []
prediction['SCF Energy'] = []

prediction['Molecular Weight'] = []
prediction['LogP'] = []
prediction['Topological Polar Surface Area'] = []
prediction['Complexity'] = []

prediction['Description'] = []
prediction['Caption'] = []

targets['HOMO'] = []
targets['LUMO'] = []
targets['HOMO-LUMO Gap'] = []
targets['SCF Energy'] = []

targets['Molecular Weight'] = []
targets['LogP'] = []
targets['Topological Polar Surface Area'] = []
targets['Complexity'] = []

targets['Description'] = []
targets['Caption'] = []

pattern = r'-?\d+\.\d+'
# pattern = r'(-?\d+\.\d+|-?\d+)'

e_prediction = []
e_target = []

for i, t in enumerate(_task_type):
    if t in ['Description', 'Caption']:
        prediction[t].append(_prediction[i])
        targets[t].append(_targets[i])
    else:
        pre_matches = re.findall(pattern, _prediction[i])
        tar_matches = re.findall(pattern, _targets[i])
        if t in ['HOMO', 'LUMO', 'HOMO-LUMO Gap']:
            if len(pre_matches) > 0 and -20 < float(pre_matches[0]) < 20:
                prediction[t].append(float(pre_matches[0]))
                targets[t].append(float(tar_matches[0]))
            else:
                e_prediction.append(_prediction[i])
                e_target.append(_targets[i])
        elif t in ['SCF Energy']:
            if len(pre_matches) > 0 and -5 < float(pre_matches[0]) < 0:
                prediction[t].append(float(pre_matches[0]))
                targets[t].append(float(tar_matches[0]))
            else:
                e_prediction.append(_prediction[i])
                e_target.append(_targets[i])
        elif t in ['LogP']:
            if len(pre_matches) > 0 and -30 < float(pre_matches[0]) < 50:
                prediction[t].append(float(pre_matches[0]))
                targets[t].append(float(tar_matches[0]))
            else:
                e_prediction.append(_prediction[i])
                e_target.append(_targets[i])
        elif t in ['Topological Polar Surface Area']:
            if len(pre_matches) > 0 and 0 <= float(pre_matches[0]) < 2000:
                prediction[t].append(float(pre_matches[0]))
                targets[t].append(float(tar_matches[0]))
            else:
                e_prediction.append(_prediction[i])
                e_target.append(_targets[i])
        elif t in ['Complexity']:
            if len(pre_matches) > 0 and 0 <= float(pre_matches[0]) < 10000:
                prediction[t].append(float(pre_matches[0]))
                targets[t].append(float(tar_matches[0]))
            else:
                e_prediction.append(_prediction[i])
                e_target.append(_targets[i])
        elif t in ['Molecular Weight']:
            if len(pre_matches) > 0 and 0 < float(pre_matches[0]) < 4000:
                prediction[t].append(float(pre_matches[0]))
                targets[t].append(float(tar_matches[0]))
            else:
                e_prediction.append(_prediction[i])
                e_target.append(_targets[i])

tokenizer = AutoTokenizer.from_pretrained('llama-2-hf-7b', use_fast=False, padding_side='right')

for key in prediction.keys():
    if key in ['Description', 'Caption']:
        print(f'{key}')
        bleu2, bleu4, rouge_1, rouge_2, rouge_l, meteor_score = caption_evaluate(prediction[key], targets[key], tokenizer, 128)
        pass
    else:
        valid_ratio = len(prediction[key])/task_type_counter[key]
        error = np.array(prediction[key]) - np.array(targets[key])
        mae = np.mean(np.abs(error))
        print(f'{key}    Valid Ratio:{valid_ratio:.4f}, MAE:{mae:.3f}')
exit()


