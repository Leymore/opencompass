from glob import glob
import json
import numpy as np

rolling_patterns = [
    'ABCD', 'ABDC', 'ACBD', 'ACDB', 'ADBC', 'ADCB',
    'BACD', 'BADC', 'BCAD', 'BCDA', 'BDAC', 'BDCA',
    'CABD', 'CADB', 'CBAD', 'CBDA', 'CDAB', 'CDBA',
    'DABC', 'DACB', 'DBAC', 'DBCA', 'DCAB', 'DCBA',
]  # ABCD -> ?
count_patterns = [str(i) for i in range(len(rolling_patterns)+1)]
content = []
headers = ["name", "rolling", "baseline", "max", "min", "std"] + count_patterns
content.append(",".join(headers))
filenames = glob('/mnt/petrelfs/zhoufengzhe/repos/pjeval/outputs/rolling_eval/4/20230718_130514/results/ChatPJLM-v0.2.2rc51-Exam-v0.1.5/lukaemon_mmlu_*.json')
filenames = sorted(filenames)
for filename in filenames:
    with open(filename, 'r') as f:
        data = json.load(f)
    name = filename.split('mmlu_')[-1].split('.json')[0]
    line = []
    line.append(name)
    line.append(data['accuracy'])
    line.append(data['pattern_accuracy']['ABCD'])
    pv = list(data['pattern_accuracy'].values())
    line.append(max(pv))
    line.append(min(pv))
    line.append(float(np.std(pv)))
    for i in count_patterns:
        line.append(data['count_accuracy'][i])

    line[1:] = [f"{x:.2f}" for x in line[1:]]
    content.append(",".join(line))


with open('/mnt/petrelfs/zhoufengzhe/repos/pjeval/scripts/rolling_eval.csv', 'w') as f:
    f.write("\n".join(content))
