from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict

def combine_dataset(dataset):
    """_summary_

    Args:
        dataset (dataframe): pandas dataframe
    """
    # Create data for custom tokenizer
    article_list = dataset['article'].tolist()
    summary_list = dataset['summary'].tolist()

    dataset_list = article_list + summary_list

    return dataset_list

def train_test(dataset):
    """_summary_

    Args:
        dataset (_type_): _description_
    """
    train_df, temp_df = train_test_split(dataset, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    return train_df, val_df, test_df
