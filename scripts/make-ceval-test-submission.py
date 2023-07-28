from glob import glob
import json
import os
import os.path as osp
from collections import Counter


def first_capital_postprocess(text: str) -> str:
    for t in text:
        if t.isupper():
            return t
    return ''


rolling_patterns = [
    'ABCD', 'ABDC', 'ACBD', 'ACDB', 'ADBC', 'ADCB',
    'BACD', 'BADC', 'BCAD', 'BCDA', 'BDAC', 'BDCA',
    'CABD', 'CADB', 'CBAD', 'CBDA', 'CDAB', 'CDBA',
    'DABC', 'DACB', 'DBAC', 'DBCA', 'DCAB', 'DCBA',
]  # ABCD -> ?

output_name2input_paths = {
    'ChatPJLM-v0.2.2rc51-Exam-v0.1.5-rolling': '/mnt/petrelfs/zhoufengzhe/repos/pjeval/outputs/rolling_eval/3/20230717_213942/predictions/ChatPJLM-v0.2.2rc51-Exam-v0.1.5',
    'ChatPJLM-v0.2.2rc51-Exam-v0.1.5': '/mnt/petrelfs/zhoufengzhe/repos/pjeval/outputs/rolling_eval/3/20230717_213942/predictions/ChatPJLM-v0.2.2rc51-Exam-v0.1.5',
}
for output_name in output_name2input_paths:
    input_paths = glob(osp.join(output_name2input_paths[output_name], "ceval-test-*.json"))
    print(osp.join(output_name2input_paths[output_name], "ceval-test-*.json"))
    overall = {}
    for input_path in input_paths:
        name = input_path.split("ceval-test-")[-1].split(".")[0]
        with open(input_path, "r") as f:
            data = json.load(f)
        results = {}
        index, base = 0, 0
        while str(base) in data:
            _results = []
            for offset, pattern in enumerate(rolling_patterns):
                if not output_name.endswith("-rolling") and offset > 0:
                    continue
                _results.append(first_capital_postprocess(data[str(base + offset)]['prediction']))
                assert _results[-1] in ["A", "B", "C", "D"]
                _results[-1] = dict(zip("ABCD", pattern))[_results[-1]]
            pred = Counter(_results).most_common(1)[0][0]
            # print(Counter(_results).most_common())
            results[str(index)] = pred
            base += len(rolling_patterns)
            index += 1

        overall[name] = results

    os.makedirs("ceval-submission", exist_ok=True)
    output_path = f"ceval-submission/{output_name}.json"
    with open(output_path, "w") as f:
        json.dump(overall, f, indent=4)
    print(output_path)
