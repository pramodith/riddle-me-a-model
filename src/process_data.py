from datasets import load_dataset, Dataset
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

load_dotenv()

def get_dataset():
    dataset = load_dataset("Pramodith/riddles_dataset_scored", token=True)["train"]
    dataset = dataset.remove_columns(["output"])
    return dataset

def get_dataset_splits(dataset: Dataset) -> tuple[Dataset, Dataset, Dataset]:
    dataset = [
        {"prompt": row["prompt"], "answer": row["answer"], "quality_score":row["quality_score"]} 
            for row in dataset
        ]
    train_dataset, dev_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
    dev_dataset, test_dataset = train_test_split(dev_dataset, test_size=0.5, random_state=42)
    return train_dataset, dev_dataset, test_dataset

def filter_dataset_on_difficulty(dataset: Dataset, difficulty: int) -> Dataset:
    return [row for row in dataset if row["quality_score"] >= difficulty]

if __name__ == "__main__":
    dataset = get_dataset()
    train_dataset, dev_dataset, test_dataset = get_dataset_splits(dataset)
    train_dataset = filter_dataset_on_difficulty(train_dataset, 2)
    dev_dataset = filter_dataset_on_difficulty(dev_dataset, 2)
    test_dataset = filter_dataset_on_difficulty(test_dataset, 2)
    print(train_dataset)
    print(dev_dataset)
    print(test_dataset)