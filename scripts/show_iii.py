from glob import glob
import json

for path in glob("/mnt/petrelfs/zhoufengzhe/repos/pjeval/outputs/InternLM_Chat/1/20230725_000210/predictions/ChatPJLM-v0.2.2rc51-Exam-v0.1.5/ceval-*.json"):
    with open(path) as f:
        data = json.load(f)
    for k, v in data.items():
        question = v["origin_prompt"].split("<|Human|>:")[-1].split("答案: െ\n<|Assistant|>:")[0].strip()
        explanation = v["prediction"].split("【解析】")[-1].split("<eoe>\n")[0].strip()
        answer = v["prediction"].split("【答案】")[-1].split("<eoa>\n")[0].strip()
        if len(answer) != 1:
            answer = '-'
        if answer != '-':
            continue
        print('-' * 128)
        print("<Q>", question, "\n<E>", explanation, "\n<A>", answer)
