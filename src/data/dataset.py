from datasets import load_dataset

# Load the databricks dataset from Hugging Face
def load_hf_dataset():
    dataset = load_dataset("conll2003",trust_remote_code=True)
    return dataset
