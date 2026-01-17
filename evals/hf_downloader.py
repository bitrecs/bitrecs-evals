import json
from datasets import load_dataset


def sample_dataset():        

    # Load in streaming mode (almost no data downloaded at start)
    #ds = load_dataset("reallybigmouse4/dense_core_amazon2023", split="train", streaming=True)
    ds = load_dataset("reallybigmouse4/fashion-eval-recsys", split="train", streaming=True)

    # Take only the first N examples (downloads ~only these examples)
    small_sample = ds.take(3)
    # You can convert to list if you want a normal python list
    small_list = list(small_sample)
    
    pretty_json = json.dumps(small_list, indent=2)
    print(pretty_json)

    count = 0
    for thing in small_list:
        #print(thing)
        count += 1

    assert count == len(small_list)

    # Or just iterate
    # for example in ds.take(3):
    #     print(example)
        # do whatever you want...   


if __name__ == "__main__":
    print("\033[33mThis is a module for downloading datasets in streaming mode. Run the file to see an example.\033[0m")
    sample_dataset()
    