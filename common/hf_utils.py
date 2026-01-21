import logging
import pandas as pd
from typing import List
from datasets import load_dataset


logger = logging.getLogger(__name__)

"""
All_Beauty
Amazon_Fashion
Appliances
Arts_Crafts_and_Sewing
Automotive
Baby_Products
Beauty_and_Personal_Care
Books
CDs_and_Vinyl
Cell_Phones_and_Accessories
Clothing_Shoes_and_Jewelry
Digital_Music
Electronics
Gift_Cards
Grocery_and_Gourmet_Food
Handmade_Products
Health_and_Household
Health_and_Personal_Care
Home_and_Kitchen
Industrial_and_Scientific
Kindle_Store
Magazine_Subscriptions
Movies_and_TV
Musical_Instruments
Office_Products
Patio_Lawn_and_Garden
Pet_Supplies
Software
Sports_and_Outdoors
Subscription_Boxes
Tools_and_Home_Improvement
Toys_and_Games
Video_Games

"""

def get_hf_folder_list(repo_id: str = "reallybigmouse4/dense_core_amazon2023") -> list:
    """
    Retrieve and print the list of top-level folders (categories) in the Hugging Face dataset.
    """
    from huggingface_hub import HfApi
    api = HfApi()
    #repo_id = "reallybigmouse4/dense_core_amazon2023"
    all_files = api.list_repo_files(repo_id, repo_type="dataset")
    top_dirs = sorted({
        f.split("/")[0]
        for f in all_files
        if "/" in f       # has at least one sub-level → it's inside a category folder
    })
    # Exclude non-category junk if present (e.g. README files, .gitattributes)
    top_dirs = [d for d in top_dirs if d not in {"README.md", ".gitattributes", ".gitignore"}]
    #print("Top-level category folders:", top_dirs)
    #print(f"Total: {len(top_dirs)}")
    top_dirs = sorted(top_dirs)  # Ensure sorted
    return top_dirs  # Return as list to preserve order


def sample_dataset(folder_name: str = "All_Beauty", size: int = 100, sample_size=5) -> pd.DataFrame:
    """
    Sample data from a specified folder and size file in the Hugging Face dataset.
    :param folder_name: Name of the folder (e.g., 'All_Beauty', 'Digital_Music')
    :param size: Size of the universe file (e.g., 100, 300, 1000)
    :param sample_size: Number of samples to take
    :return: DataFrame of sampled data
    """
    # Construct the file name based on folder and size
    file_name = f"dense_core_amazon_{folder_name.lower().replace('_', '_')}_universe_{size}.csv"
    dataset_file = f"https://huggingface.co/datasets/reallybigmouse4/dense_core_amazon2023/resolve/main/{folder_name}/{file_name}"
    try:
        ds = load_dataset(
            "csv",
            data_files={"train": dataset_file},
            split="train",
            streaming=True,  
            column_names=['created_at', 'query', 'ground_truth_sku', 'batch_id', 'model', 'provider', 'winning_response', 'context'],
            skiprows=1  # Skip the header row in the CSV file
        )
        print(f"columns = {ds.column_names}")
        small_sample = ds.take(sample_size)
        small_list = list(small_sample)
        df = pd.DataFrame(small_list)
        # Ensure columns align with expected names
        expected_columns = ['created_at', 'query', 'ground_truth_sku', 'batch_id', 'model', 'provider', 'winning_response', 'context']
        if list(df.columns) != expected_columns:          
            raise ValueError(f"Column mismatch: expected {expected_columns}, got {list(df.columns)}.")
        return df
    except Exception as e:
        logger.error(f"Failed to load dataset from '{dataset_file}': {e}")
        return pd.DataFrame()

