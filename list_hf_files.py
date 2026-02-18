from huggingface_hub import HfApi
try:
    api = HfApi()
    files = api.list_repo_files("reallybigmouse4/ndgc_amazon_curated", repo_type="dataset")
    print(files)
except Exception as e:
    print(e)
