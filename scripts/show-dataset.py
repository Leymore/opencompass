# from opencompass.datasets import CEvalRobustDataset
# dataset = CEvalRobustDataset.load("./data/ceval/formal_ceval", 'computer_network')
# print(dataset["val"][:4])

from opencompass.datasets import MMLURobustDataset
dataset = MMLURobustDataset.load("./data/mmlu", 'college_biology')
print(dataset["test"][:4])

from IPython import embed; embed()
