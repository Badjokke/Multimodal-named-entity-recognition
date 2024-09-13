import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset


def load_news_dataset(path="news_text.csv"):
    df = pd.read_csv(path,sep="\t")
    df = df.dropna(subset=["abstract"])
    df["input_prompt"] = ("TITLE: " + df["title"] + "\n" + "ABSTRACT: " + df["abstract"])
    train_en, test_en = train_test_split(df, test_size = 0.3, random_state=42)
    print("Train Data Shape:", train_en.shape)
    print("Test Data Shape:", test_en.shape)
    return train_en, test_en


# Load the databricks dataset from Hugging Face
def load_hf_dataset():
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
    print(f'Number of prompts: {len(dataset)}')
    print(f'Column names are: {dataset.column_names}')
    return dataset
    