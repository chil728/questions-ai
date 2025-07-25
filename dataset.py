from datasets import load_dataset, Dataset, DatasetDict
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from random import randint

DATASET_NAME = ""

dataset = load_dataset(path=DATASET_NAME)

df = pd.DataFrame(dataset['train'])

df = df.iloc[:]

train, test = train_test_split(df, test_size=0.1, random_state=666)

print(f"Training Samples Amount: {len(train)}")
print(f"Testing Samples Amount: {len(test)}")

prompt_template = """


<|Question|>
{question}

<|Answer|>
{answer}
"""

def template_dataset(sample):
    sample['text'] = prompt_template.format(
        question = sample["question"],
        answer = sample['answer']
    )
    return sample

train_dataset = Dataset.from_pandas(df=train)
test_dataset = Dataset.from_pandas(df=test)

dataset = DatasetDict(
    "train": train_dataset,
    "test": test_dataset,
)

train_dataset = dataset['train'].map(
    function=template_dataset,
    remove_columns=list(dataset['train'].features)
)

test_dataset = dataset['test'].map(
    function=template_dataset,
    remove_columns=list(dataset['test'].features)
)

os.makedirs(name="data", exist_ok=True)
train_dataset.to_json(path_or_buf="data/train.jsonl")
test_dataset.to_json(path_or_buf="data/test.jsonl")
