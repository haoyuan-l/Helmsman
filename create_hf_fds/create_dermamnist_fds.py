import os
import argparse
from tqdm import tqdm
from datasets import Dataset, DatasetDict, Features, ClassLabel, Image
from huggingface_hub import HfApi, login
from medmnist.dataset import DermaMNIST

def load_dermamnist_data(data_path: str):
    """
    Load DermaMNIST dataset from the medmnist library and transform it
    into a Hugging Face DatasetDict.
    
    Args:
        data_path (str): Path to the directory where the dataset will be downloaded.
    
    Returns:
        tuple: A tuple containing:
            - DatasetDict: The Hugging Face dataset with 'train' and 'test' splits.
            - dict: A dictionary containing information about the dataset labels.
    """
    
    # Get dataset information from the medmnist library
    print("Loading 'train' split from medmnist to get dataset info...")
    train_medmnist = DermaMNIST(root=data_path, split='train', download=True)
    info = train_medmnist.info
    label_map = info['label']
    label_names = list(label_map.values())
    
    print(f"Dataset: DermaMNIST")
    print(f"Description: {info['description']}")
    print(f"Number of classes: {len(label_names)}")
    print(f"Labels: {label_names}")

    def create_hf_dataset_from_split(split: str):
        """Load a specific split of DermaMNIST and convert it to a Hugging Face Dataset."""
        print(f"Loading '{split}' split...")
        # We load the dataset without any transforms to upload the raw PIL images
        medmnist_dataset = DermaMNIST(root=data_path, split=split, download=True)
        
        # Create a dictionary from the dataset
        # We use tqdm for a nice progress bar
        data_dict = {'image': [], 'label': [], 'label_name': []}
        
        # Squeeze the labels to be 1D
        labels = medmnist_dataset.labels.squeeze()
        
        for i in tqdm(range(len(medmnist_dataset)), desc=f"Processing {split} images"):
            image, label_tuple = medmnist_dataset[i]
            label_int = label_tuple[0]

            data_dict['image'].append(image)
            data_dict['label'].append(label_int)
            data_dict['label_name'].append(label_names[label_int])
            
        return Dataset.from_dict(data_dict)

    # Create datasets for train and test splits
    train_dataset = create_hf_dataset_from_split('train')
    test_dataset = create_hf_dataset_from_split('test')

    # Define the features for the dataset
    features = Features({
        'image': Image(),
        'label': ClassLabel(names=label_names),
        'label_name': ClassLabel(names=label_names) # Storing string label for convenience
    })
    
    # Set the features for each dataset
    train_dataset = train_dataset.cast(features)
    test_dataset = test_dataset.cast(features)
    
    # Create the final DatasetDict object
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })
    
    return dataset_dict, info

def create_dataset_card(dataset_info: dict, train_samples: int, test_samples: int, num_clients_example: int = 10):
    """
    Create a comprehensive dataset card (README.md) for the Hugging Face Hub.
    
    Args:
        dataset_info (dict): MedMNIST info dictionary.
        train_samples (int): Number of samples in the training set.
        test_samples (int): Number of samples in the test set.
        num_clients_example (int): Example number of clients for FL partitioning.
    
    Returns:
        str: The content of the README.md file as a string.
    """
    
    label_list_md = "\n".join([f"- **{idx}**: `{name}`" for idx, name in dataset_info['label'].items()])
    
    card_content = f"""---
    license: cc-by-nc-4.0
    task_categories:
    - image-classification
    tags:
    - medmnist
    - medical-imaging
    - skin-cancer
    - federated-learning
    - healthcare
    size_categories:
    - 10K<n<100K
    ---

    # DermaMNIST Dataset for Federated Learning

    ## Dataset Description

    **DermaMNIST** is a multi-class dataset of dermatoscopic images for skin lesion classification. It is part of the [MedMNIST v2](https://medmnist.com/) collection, a set of standardized biomedical image datasets for machine learning. This version of the dataset has been prepared for easy use in federated learning simulations, particularly with the [Flower](https://flower.dev/) framework.

    The dataset is based on the [HAM10000](https://doi.org/10.1038/sdata.2018.161) dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. The images are 28x28 pixels and resized from their original 600x450 resolution.

    ### Classes
    The dataset contains 7 classes of skin lesions:
    {label_list_md}

    ### Dataset Statistics
    - **Number of classes**: {len(dataset_info['label'])}
    - **Training samples**: {train_samples:,}
    - **Test samples**: {test_samples:,}
    - **Total samples**: {train_samples + test_samples:,}

    ## Dataset Structure

    ### Data Fields
    - `image`: A PIL Image object of size 28x28.
    - `label`: The integer class ID.
    - `label_name`: The string name of the class label.

    ### Data Splits
    - **Train**: {train_samples:,} samples
    - **Test**: {test_samples:,} samples

    ## Usage with Flower Datasets 🌸

    This dataset is ideal for federated learning simulations where data is distributed across multiple clients (e.g., hospitals). Here's how to use it with [Flower Datasets](https://flower.dev/docs/datasets/):

    ```python
    from flwr_datasets import FederatedDataset

    # Define the number of clients (e.g., simulating 10 hospitals)
    NUM_CLIENTS = {num_clients_example}

    # Load the dataset and partition it into {num_clients_example} clients
    fds = FederatedDataset(dataset="your-username/dermamnist-fl", partitioners={{"train": NUM_CLIENTS}})

    # Get the data for a specific client
    client_data = fds.load_partition(0, "train")

    # You can now use this client_data with your standard PyTorch or TensorFlow data loaders.
    # For example, to set the format for PyTorch:
    # client_data.set_format(type="torch", columns=["image", "label"])
    ```
    ## Citation
    If you use this dataset in your research, please cite the following paper:

    ```
    @article{{medmnistv2,
        title={{MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification}},
        author={{Jiancheng Yang and Rui Shi and Donglai Wei and Zequan Liu and Lin Zhao and Bilian Ke and Hanspeter Pfister and Bingbing Ni}},
        journal={{Scientific Data}},
        year={{2023}},
        volume={{10}},
        number={{1}},
        pages={{41}},
        doi={{10.1038/s41597-022-01721-8}}
    }}
    ```
    ## Citation
    ```bibtex
        @article{{medmnistv2,
            title={{MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification}},
            author={{Jiancheng Yang and Rui Shi and Donglai Wei and Zequan Liu and Lin Zhao and Bilian Ke and Hanspeter Pfister and Bingbing Ni}},
            journal={{Scientific Data}},
            year={{2023}},
            volume={{10}},
            number={{1}},
            pages={{41}},
            doi={{10.1038/s41597-022-01721-8}}
        }}
    ```
    ## Example Usage with datasets
    ```python
    from datasets import load_dataset
    from PIL import Image

    # Load the dataset
    dataset = load_dataset("your-username/dermamnist-fl")

    # Access the training split
    train_data = dataset["train"]
    print(f"Number of training samples: {{len(train_data)}}")

    # Access a sample
    sample = train_data[100]
    image: Image = sample['image']
    print(f"Image size: {{image.size}}")
    print(f"Label: {{sample['label']}}")
    print(f"Label name: {{sample['label_name']}}")

    # To display the image
    # image.show()
    ```
    """

    return card_content

def main():
    parser = argparse.ArgumentParser(description='Upload DermaMNIST dataset to HuggingFace Hub for Federated Learning')
    parser.add_argument('--data_path', type=str, default='./data',
        help='Path to directory for downloading the dataset')
    parser.add_argument('--dataset_name', type=str, default="dermamnist-fl",
        help='Name for the dataset on HuggingFace Hub (e.g., "dermamnist-fl")')
    parser.add_argument('--hf_username', type=str, required=True,
        help='Your HuggingFace username')
    parser.add_argument('--private', action='store_true',
        help='Make the dataset private on HuggingFace Hub')
    
    args = parser.parse_args()

    # --- 1. Authenticate with Hugging Face ---
    print("Logging in to Hugging Face Hub...")
    try:
        login()
        print("✅ Successfully authenticated with HuggingFace Hub")
    except Exception as e:
        print(f"❌ Login failed: {e}")
        print("Please run 'huggingface-cli login' from your terminal and enter your token.")
        return

    # --- 2. Load and Process Data ---
    print(f"Loading DermaMNIST dataset from medmnist library...")
    try:
        dataset_dict, dataset_info = load_dermamnist_data(args.data_path)
        
        print("\n✅ Dataset loaded and processed successfully!")
        print(f"📊 Train samples: {len(dataset_dict['train']):,}")
        print(f"📊 Test samples: {len(dataset_dict['test']):,}")
        
        sample = dataset_dict['train'][0]
        print(f"\n📋 Sample data:")
        print(f"   Image type: {type(sample['image'])}")
        print(f"   Image size: {sample['image'].size}")
        print(f"   Label: {sample['label']}")
        print(f"   Label Name: {sample['label_name']}")
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return

    # --- 3. Upload Dataset to Hub ---
    repo_name = f"{args.hf_username}/{args.dataset_name}"
    print(f"\n🚀 Uploading dataset to Hugging Face Hub: {repo_name}")

    try:
        dataset_dict.push_to_hub(
            repo_id=repo_name,
            private=args.private,
            commit_message="Initial upload of DermaMNIST dataset for federated learning"
        )
        print(f"✅ Dataset successfully uploaded!")
        
        # --- 4. Create and Upload Dataset Card (README.md) ---
        print("📝 Creating and uploading dataset card...")
        card_content = create_dataset_card(
            dataset_info,
            len(dataset_dict['train']),
            len(dataset_dict['test'])
        )
        
        api = HfApi()
        api.upload_file(
            path_or_fileobj=card_content.encode(),
            path_in_repo="README.md",
            repo_id=repo_name,
            repo_type="dataset",
            commit_message="Add comprehensive dataset card with FL usage instructions"
        )
        
        print("✅ Dataset card uploaded!")
        
    except Exception as e:
        print(f"❌ Error uploading to HuggingFace Hub: {e}")
        return
    # --- 5. Final Instructions ---
    print(f"\n🎉 All done! Your dataset is live at: https://huggingface.co/datasets/{repo_name}")
    print("\n📋 Next steps:")
    print("1. Visit your dataset page and check the README and data viewer.")
    print(f"2. Test loading your dataset: `from datasets import load_dataset; ds = load_dataset('{repo_name}')`")
    print("3. Test with Flower Datasets:")
    print(f"   `from flwr_datasets import FederatedDataset`")
    print(f"   `fds = FederatedDataset(dataset='{repo_name}', partitioners={{'train': 10}})`")
    print(f"   `partition = fds.load_partition(0, 'train')`")

if __name__ == "__main__":
    main()