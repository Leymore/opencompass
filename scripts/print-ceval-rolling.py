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
filenames = glob('/mnt/petrelfs/zhoufengzhe/repos/pjeval/outputs/rolling_eval/3/20230717_213942/results/ChatPJLM-v0.2.2rc51-Exam-v0.1.5/ceval-*.json')
filenames = sorted(filenames)
for filename in filenames:
    with open(filename, 'r') as f:
        data = json.load(f)
    # {"accuracy": 41.37931034482759, "pattern_accuracy": {"ABCD": 65.51724137931035, "ABDC": 62.06896551724138, "ACBD": 62.06896551724138, "ACDB": 58.620689655172406, "ADBC": 65.51724137931035, "ADCB": 62.06896551724138, "BACD": 58.620689655172406, "BADC": 58.620689655172406, "BCAD": 65.51724137931035, "BCDA": 58.620689655172406, "BDAC": 58.620689655172406, "BDCA": 58.620689655172406, "CABD": 72.41379310344827, "CADB": 62.06896551724138, "CBAD": 65.51724137931035, "CBDA": 58.620689655172406, "CDAB": 65.51724137931035, "CDBA": 65.51724137931035, "DABC": 58.620689655172406, "DACB": 62.06896551724138, "DBAC": 62.06896551724138, "DBCA": 62.06896551724138, "DCAB": 62.06896551724138, "DCBA": 65.51724137931035}, "count_accuracy": {"0": 100.0, "1": 89.65517241379311, "2": 86.20689655172413, "3": 82.75862068965517, "4": 82.75862068965517, "5": 82.75862068965517, "6": 79.3103448275862, "7": 72.41379310344827, "8": 72.41379310344827, "9": 68.96551724137932, "10": 65.51724137931035, "11": 58.620689655172406, "12": 58.620689655172406, "13": 58.620689655172406, "14": 58.620689655172406, "15": 55.172413793103445, "16": 48.275862068965516, "17": 48.275862068965516, "18": 48.275862068965516, "19": 48.275862068965516, "20": 48.275862068965516, "21": 48.275862068965516, "22": 48.275862068965516, "23": 44.827586206896555, "24": 41.37931034482759}}
    name = filename.split('ceval-')[-1].split('.json')[0]
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

with open('rolling_eval.csv', 'w') as f:
    f.write("\n".join(content))
