import os
import numpy as np
from datasets import Dataset, DatasetDict, Features, Value, Image
from huggingface_hub import HfApi, login
import argparse
from datasets import load_dataset

def create_longtail_cifar10(imbalance_factor=100):
    """
    Create CIFAR-10 Long-Tail dataset with exponential class imbalance
    
    Args:
        imbalance_factor (int): Ratio between head and tail classes (default: 100)
    
    Returns:
        DatasetDict: HuggingFace dataset with train and test splits
    """
    
    # CIFAR-10 class names
    class_names = [
        'airplane', 'automobile', 'bird', 'cat', 'deer', 
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
    
    print(f"Loading original CIFAR-10 dataset...")
    # Load the original CIFAR-10 dataset
    cifar10 = load_dataset("uoft-cs/cifar10")
    train_data = cifar10["train"]
    test_data = cifar10["test"]
    
    print(f"Original CIFAR-10 train samples: {len(train_data)}")
    print(f"Original CIFAR-10 test samples: {len(test_data)}")
    
    # Create long-tailed training set
    print(f"\nCreating long-tail distribution with imbalance factor {imbalance_factor}:")
    
    longtail_indices = []
    class_sample_counts = {}
    
    for class_id in range(10):  # CIFAR-10 has 10 classes
        # Calculate samples per class using exponential decay
        if class_id == 0:
            samples_for_class = 5000  # Head class (maximum in CIFAR-10)
        else:
            # Exponential decay from head to tail
            decay_factor = (imbalance_factor) ** (class_id / 9.0)
            samples_for_class = max(int(5000 / decay_factor), 5000 // imbalance_factor)
        
        # Get indices for this class
        class_indices = [i for i, label in enumerate(train_data["label"]) if label == class_id]
        
        # Sample the required number of indices
        if len(class_indices) >= samples_for_class:
            selected_indices = class_indices[:samples_for_class]
        else:
            selected_indices = class_indices
        
        longtail_indices.extend(selected_indices)
        class_sample_counts[class_id] = len(selected_indices)
        
        print(f"  Class {class_id} ({class_names[class_id]}): {len(selected_indices)} samples")
    
    # Create long-tailed training dataset
    longtail_train = train_data.select(longtail_indices)
    
    # Keep test set unchanged (balanced for fair evaluation)
    longtail_test = test_data
    
    print(f"\nLong-tail dataset created:")
    print(f"  Train samples: {len(longtail_train)}")
    print(f"  Test samples: {len(longtail_test)}")
    print(f"  Imbalance ratio: {max(class_sample_counts.values())}:{min(class_sample_counts.values())}")
    
    # Create DatasetDict
    dataset_dict = DatasetDict({
        'train': longtail_train,
        'test': longtail_test
    })
    
    return dataset_dict, class_sample_counts, class_names

def create_dataset_card(class_sample_counts, class_names, train_samples, test_samples, imbalance_factor):
    """Create a comprehensive dataset card for HuggingFace Hub"""
    
    # Calculate statistics
    total_train_samples = sum(class_sample_counts.values())
    head_samples = max(class_sample_counts.values())
    tail_samples = min(class_sample_counts.values())
    actual_imbalance = head_samples / tail_samples if tail_samples > 0 else head_samples
    
    card_content = f"""---
license: mit
task_categories:
- image-classification
tags:
- computer-vision
- long-tail-classification
- class-imbalance
- federated-learning
- cifar-10
size_categories:
- 10K<n<100K
---

# CIFAR-10 Long-Tail Federated Dataset

## Dataset Description

This is a long-tailed version of CIFAR-10 designed for federated learning research. The dataset introduces class imbalance following an exponential decay distribution, making it ideal for studying long-tail classification in federated settings.

### Class Distribution (Training Set)
The training set follows a long-tail distribution with imbalance factor {imbalance_factor}:

{chr(10).join([f"- **{class_names[i]}** (Class {i}): {class_sample_counts[i]:,} samples" for i in range(10)])}

**Actual Imbalance Ratio**: {actual_imbalance:.1f}:1 (Head:Tail)

### Dataset Statistics
- **Number of classes**: 10
- **Training samples**: {total_train_samples:,} (long-tail distribution)
- **Test samples**: {test_samples:,} (balanced distribution)
- **Image size**: 32x32 pixels
- **Channels**: 3 (RGB)
- **Head class samples**: {head_samples:,}
- **Tail class samples**: {tail_samples:,}

## Dataset Structure

### Data Fields
- `img`: PIL Image object (32x32x3)
- `label`: Integer class label (0-9)

### Data Splits
- **Train**: {total_train_samples:,} samples (long-tail distributed)
- **Test**: {test_samples:,} samples (balanced for fair evaluation)

## Class Labels
{chr(10).join([f"{i}: {name}" for i, name in enumerate(class_names)])}

## Usage with Flower Datasets

This dataset is optimized for federated learning research on long-tail classification:

```python
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner

# Load the dataset with IID partitioning across clients
partitioner = IidPartitioner(num_partitions=10)
fds = FederatedDataset(
    dataset="your-username/cifar10-lt-federated", 
    partitioners={{"train": partitioner}}
)

# Get data for a specific client
client_data = fds.load_partition(0)
print(f"Client 0 has {{len(client_data)}} samples")
```

### Federated Learning Scenarios

This dataset supports several FL research scenarios:

1. **Long-tail FL**: Study how federated learning handles class imbalance
2. **IID distribution**: Each client gets similar long-tail distribution  
3. **Non-IID variants**: Can be combined with other partitioners for heterogeneous settings
4. **Fairness research**: Analyze performance across head vs tail classes

## Comparison with Standard CIFAR-10

| Metric | Standard CIFAR-10 | CIFAR-10-LT (This Dataset) |
|--------|------------------|---------------------------|
| Train samples per class | 5,000 (uniform) | {head_samples:,} → {tail_samples} (exponential decay) |
| Test samples per class | 1,000 (uniform) | 1,000 (uniform, unchanged) |
| Total train samples | 50,000 | {total_train_samples:,} |
| Class distribution | Balanced | Long-tail (IF={imbalance_factor}) |

## Research Applications

This dataset is particularly useful for:

- **Long-tail classification research**: Studying methods to handle class imbalance
- **Federated learning with imbalanced data**: Real-world FL scenarios
- **Fairness in ML**: Analyzing model performance across different class frequencies  
- **Few-shot learning**: Tail classes have limited samples
- **Meta-learning**: Learning from classes with varying sample sizes

## Example Usage

```python
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
dataset = load_dataset("your-username/cifar10-lt-federated")

# Analyze class distribution
train_data = dataset["train"]
labels = train_data["label"]
class_counts = np.bincount(labels)

print("Class distribution:")
class_names = {class_names}
for i, count in enumerate(class_counts):
    print(f"{{class_names[i]}}: {{count}} samples")

# Visualize class distribution
plt.figure(figsize=(10, 6))
plt.bar(range(10), class_counts)
plt.xlabel("Class")
plt.ylabel("Number of Samples")
plt.title("CIFAR-10 Long-Tail Class Distribution")
plt.xticks(range(10), [name[:4] for name in class_names], rotation=45)
plt.yscale('log')
plt.show()

# Access a sample
sample = train_data[0]
print(f"Image shape: {{np.array(sample['img']).shape}}")
print(f"Label: {{sample['label']}} ({{class_names[sample['label']]}})")
```

## Performance Baselines

Due to the long-tail distribution, standard classification methods typically show:
- High accuracy on head classes (airplane: ~90%+)
- Poor accuracy on tail classes (truck: ~40-60%)
- Overall accuracy drop compared to balanced CIFAR-10

This creates opportunities for research on:
- Loss reweighting methods
- Data augmentation for tail classes
- Two-stage training approaches
- Ensemble methods

## Citation

```bibtex
@misc{{cifar10-lt-federated,
  title={{CIFAR-10 Long-Tail Federated Dataset}},
  author={{Created for Federated Long-Tail Classification Research}},
  year={{2024}},
  url={{https://huggingface.co/datasets/your-username/cifar10-lt-federated}}
}}
```

## Original CIFAR-10 Citation

```bibtex
@techreport{{krizhevsky2009learning,
  title={{Learning multiple layers of features from tiny images}},
  author={{Krizhevsky, Alex and Hinton, Geoffrey and others}},
  year={{2009}},
  institution={{Technical report, University of Toronto}}
}}
```

## License

MIT License - Same as original CIFAR-10 dataset

## Related Work

- [Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss](https://arxiv.org/abs/1906.07413)
- [Long-tail Recognition by Routing Diverse Distribution-Aware Experts](https://arxiv.org/abs/2010.01809)
- [Decoupling Representation and Classifier for Long-Tailed Recognition](https://arxiv.org/abs/1910.09217)
"""
    
    return card_content

def main():
    parser = argparse.ArgumentParser(description='Create and upload CIFAR-10 Long-Tail federated dataset to HuggingFace Hub')
    parser.add_argument('--imbalance_factor', type=int, default=100,
                      help='Imbalance factor between head and tail classes (default: 100)')
    parser.add_argument('--dataset_name', type=str, default='cifar10-lt-federated',
                      help='Name for the dataset on HuggingFace Hub')
    parser.add_argument('--hf_username', type=str, required=True,
                      help='Your HuggingFace username')
    parser.add_argument('--private', action='store_true',
                      help='Make the dataset private on HuggingFace Hub')
    parser.add_argument('--no_upload', action='store_true',
                      help='Create dataset locally without uploading (for testing)')
    
    args = parser.parse_args()
    
    if not args.no_upload:
        # Authenticate with HuggingFace
        print("Please make sure you're logged in to HuggingFace Hub...")
        print("Run 'huggingface-cli login' if you haven't already")
        
        try:
            login()
            print("✅ Successfully authenticated with HuggingFace Hub")
        except Exception as e:
            print(f"❌ Login failed: {e}")
            print("Please run 'huggingface-cli login' first")
            return
    
    print(f"Creating CIFAR-10 Long-Tail dataset with imbalance factor {args.imbalance_factor}...")
    
    try:
        # Create the long-tail dataset
        dataset_dict, class_sample_counts, class_names = create_longtail_cifar10(args.imbalance_factor)
        
        print(f"✅ Long-tail dataset created successfully!")
        print(f"📊 Train samples: {len(dataset_dict['train']):,}")
        print(f"📊 Test samples: {len(dataset_dict['test']):,}")
        
        # Show sample data
        sample = dataset_dict['train'][0]
        print(f"\n📋 Sample data:")
        print(f"   Image shape: {np.array(sample['img']).shape}")
        print(f"   Label: {sample['label']} ({class_names[sample['label']]})")
        
        # Show class distribution
        print(f"\n📊 Class distribution summary:")
        head_samples = max(class_sample_counts.values())
        tail_samples = min(class_sample_counts.values())
        print(f"   Head class: {head_samples:,} samples")
        print(f"   Tail class: {tail_samples:,} samples")
        print(f"   Imbalance ratio: {head_samples/tail_samples:.1f}:1")
        
    except Exception as e:
        print(f"❌ Error creating dataset: {e}")
        return
    
    if args.no_upload:
        print("📁 Dataset created locally (no upload requested)")
        return
    
    # Create repository name
    repo_name = f"{args.hf_username}/{args.dataset_name}"
    
    print(f"\n🚀 Uploading dataset to: {repo_name}")
    
    try:
        # Push to HuggingFace Hub
        dataset_dict.push_to_hub(
            repo_id=repo_name,
            private=args.private,
            commit_message=f"Upload CIFAR-10 Long-Tail dataset with imbalance factor {args.imbalance_factor}"
        )
        
        print(f"✅ Dataset successfully uploaded!")
        
        # Create and upload dataset card
        card_content = create_dataset_card(
            class_sample_counts,
            class_names,
            len(dataset_dict['train']),
            len(dataset_dict['test']),
            args.imbalance_factor
        )
        
        api = HfApi()
        api.upload_file(
            path_or_fileobj=card_content.encode(),
            path_in_repo="README.md",
            repo_id=repo_name,
            repo_type="dataset",
            commit_message="Add comprehensive dataset card for CIFAR-10 Long-Tail FL"
        )
        
        print(f"✅ Dataset card uploaded!")
        
    except Exception as e:
        print(f"❌ Error uploading to HuggingFace Hub: {e}")
        return
    
    print(f"\n🎉 Dataset successfully uploaded to: https://huggingface.co/datasets/{repo_name}")
    print("\n📋 Next steps:")
    print("1. Visit your dataset page and review the README")
    print(f"2. Test loading the dataset:")
    print(f"   from datasets import load_dataset")
    print(f"   dataset = load_dataset('{repo_name}')")
    print("3. Test with Flower Datasets:")
    print(f"   from flwr_datasets import FederatedDataset")
    print(f"   from flwr_datasets.partitioner import IidPartitioner")
    print(f"   partitioner = IidPartitioner(num_partitions=10)")
    print(f"   fds = FederatedDataset(dataset='{repo_name}', partitioners={{'train': partitioner}})")
    print("4. Use in your federated learning experiments!")
    
    # Save class distribution info locally
    print(f"\n💾 Saving class distribution info to 'cifar10_lt_class_distribution.txt'")
    with open('cifar10_lt_class_distribution.txt', 'w') as f:
        f.write(f"CIFAR-10 Long-Tail Class Distribution (IF={args.imbalance_factor})\n")
        f.write("=" * 50 + "\n")
        for i in range(10):
            f.write(f"Class {i} ({class_names[i]}): {class_sample_counts[i]:,} samples\n")
        f.write(f"\nTotal train samples: {sum(class_sample_counts.values()):,}\n")
        f.write(f"Imbalance ratio: {head_samples/tail_samples:.1f}:1\n")

if __name__ == "__main__":
    main()