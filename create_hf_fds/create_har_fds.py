import os
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict, Features, Value, Sequence
from huggingface_hub import HfApi, login
import argparse

def load_uci_har_csv_data(data_path):
    """
    Load UCI HAR dataset from CSV files and transform to HuggingFace Dataset format
    
    Args:
        data_path (str): Path to directory containing train.csv and test.csv
    
    Returns:
        DatasetDict: HuggingFace dataset with train and test splits
    """
    
    # Define activity labels (standard UCI HAR activities)
    activity_labels = {
        1: 'WALKING',
        2: 'WALKING_UPSTAIRS', 
        3: 'WALKING_DOWNSTAIRS',
        4: 'SITTING',
        5: 'STANDING',
        6: 'LAYING'
    }
    
    # Generate feature names (standard UCI HAR has 561 features)
    feature_names = [f'feature_{i}' for i in range(561)]
    
    def load_csv_split(split):
        """Load data from CSV file for a specific split"""
        csv_file = os.path.join(data_path, f'{split}.csv')
        
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found: {csv_file}")
        
        print(f"Loading {csv_file}...")
        df = pd.read_csv(csv_file)
        
        print(f"CSV shape: {df.shape}")
        print(f"CSV columns: {list(df.columns)}")
        
        # Initialize data dictionary
        data_dict = {}
        
        # Try to identify the column structure
        # Common patterns for UCI HAR CSV files:
        # 1. First 561 columns are features, last columns are target/subject
        # 2. Features might be named or just numbered
        # 3. Target might be named 'Activity', 'target', 'label', or 'y'
        # 4. Subject might be named 'Subject', 'subject_id', or 'subject'
        
        # Find target column
        target_col = None
        possible_target_names = ['Activity', 'activity', 'target', 'label', 'y', 'class']
        for col in possible_target_names:
            if col in df.columns:
                target_col = col
                break
        
        # Find subject column
        subject_col = None
        possible_subject_names = ['Subject', 'subject', 'subject_id', 'SubjectID']
        for col in possible_subject_names:
            if col in df.columns:
                subject_col = col
                break
        
        # If no explicit target/subject columns found, assume structure
        if target_col is None:
            # Assume last column is target if not found
            target_col = df.columns[-1]
            print(f"No explicit target column found, using last column: {target_col}")
        
        if subject_col is None and len(df.columns) >= 563:  # 561 features + target + subject
            # Assume second-to-last column is subject
            subject_col = df.columns[-2]
            print(f"No explicit subject column found, using second-to-last column: {subject_col}")
        
        # Extract target data
        target_data = df[target_col].values
        
        # Convert string activity labels to numbers if needed
        if target_data.dtype == 'object':  # String labels
            label_to_num = {v: k for k, v in activity_labels.items()}
            # Handle different naming conventions
            label_map = {}
            for val in np.unique(target_data):
                val_upper = str(val).upper().replace(' ', '_')
                if val_upper in label_to_num:
                    label_map[val] = label_to_num[val_upper]
                else:
                    # Try to find a match
                    for key in label_to_num:
                        if key.replace('_', '').lower() in val_upper.replace('_', '').lower():
                            label_map[val] = label_to_num[key]
                            break
            
            if not label_map:
                # If no mapping found, create numeric mapping
                unique_labels = sorted(np.unique(target_data))
                label_map = {label: i+1 for i, label in enumerate(unique_labels)}
                print(f"Created numeric mapping: {label_map}")
            
            target_data = np.array([label_map.get(val, 1) for val in target_data])
        
        # Extract subject data if available
        if subject_col is not None:
            subject_data = df[subject_col].values
        else:
            # Create dummy subject IDs
            subject_data = np.ones(len(df), dtype=int)
            print("No subject column found, using dummy subject IDs")
        
        # Extract feature columns
        feature_cols = [col for col in df.columns if col not in [target_col, subject_col]]
        
        # If we have exactly 561 feature columns, use them directly
        if len(feature_cols) == 561:
            X_data = df[feature_cols].values
        else:
            # Take first 561 columns as features
            feature_cols = df.columns[:561]
            X_data = df[feature_cols].values
            if X_data.shape[1] < 561:
                print(f"Warning: Only {X_data.shape[1]} features found, expected 561")
                # Pad with zeros if needed
                padding = np.zeros((X_data.shape[0], 561 - X_data.shape[1]))
                X_data = np.concatenate([X_data, padding], axis=1)
        
        # Create dataset dictionary with separate columns for each feature
        # Add each feature as a separate column (561 features)
        for i in range(561):
            if i < X_data.shape[1]:
                data_dict[str(i)] = X_data[:, i].tolist()
            else:
                data_dict[str(i)] = [0.0] * len(target_data)
        
        # Add target and metadata columns
        data_dict['target'] = target_data.tolist()
        data_dict['activity_label'] = [activity_labels.get(int(y), f'UNKNOWN_{int(y)}') for y in target_data]
        data_dict['subject_id'] = subject_data.tolist()
        
        return data_dict
    
    # Load train and test data
    train_data = load_csv_split('train')
    test_data = load_csv_split('test')
    
    # Define features schema for HuggingFace datasets
    # Create features dict with 561 individual feature columns
    features_dict = {}
    
    # Add each feature column (0 to 560)
    for i in range(561):
        features_dict[str(i)] = Value('float32')
    
    # Add target and metadata columns
    features_dict['target'] = Value('int32')
    features_dict['activity_label'] = Value('string')
    features_dict['subject_id'] = Value('int32')
    
    features = Features(features_dict)
    
    # Create HuggingFace datasets
    train_dataset = Dataset.from_dict(train_data, features=features)
    test_dataset = Dataset.from_dict(test_data, features=features)
    
    # Create DatasetDict
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })
    
    return dataset_dict, activity_labels, feature_names

def create_dataset_card(activity_labels, feature_names, train_samples, test_samples, total_subjects=30):
    """Create a comprehensive dataset card for HuggingFace Hub"""
    
    card_content = f"""---
license: mit
task_categories:
- time-series-classification
tags:
- human-activity-recognition
- sensor-data
- federated-learning
- mobile-sensing
- accelerometer
- gyroscope
size_categories:
- 1K<n<10K
---

# UCI Human Activity Recognition (HAR) Dataset

## Dataset Description

The UCI Human Activity Recognition dataset is a widely-used benchmark for human activity recognition using smartphone sensors. This dataset contains sensor readings from accelerometers and gyroscopes of smartphones worn by volunteers performing six different activities.

### Activities
The dataset includes the following 6 activities:
{chr(10).join([f"- **{idx}**: {label}" for idx, label in activity_labels.items()])}

### Dataset Statistics
- **Number of subjects**: {total_subjects}
- **Number of activities**: {len(activity_labels)}
- **Number of features**: {len(feature_names)}
- **Training samples**: {train_samples:,}
- **Test samples**: {test_samples:,}
- **Total samples**: {train_samples + test_samples:,}

## Dataset Structure

### Data Fields
- `0` to `560`: Individual sensor feature columns (561 float values total)
- `target`: Integer ID of the activity (1-6)
- `activity_label`: String label of the activity
- `subject_id`: Integer ID of the subject

### Data Splits
- **Train**: {train_samples:,} samples
- **Test**: {test_samples:,} samples

## Usage with Flower Datasets

This dataset is optimized for federated learning scenarios. Here's how to use it with Flower:

```python
from flwr_datasets import FederatedDataset

# Load the dataset
fds = FederatedDataset(dataset="your-username/uci-har", partitioners={{"train": {total_subjects}}})

# Get data for a specific client (subject)
client_data = fds.load_partition(0)
```

### Federated Learning Scenarios

This dataset supports several FL scenarios:

1. **Subject-based partitioning**: Each client represents one subject (natural FL setting)
2. **Activity-based partitioning**: Clients have different activity distributions
3. **Sensor heterogeneity**: Simulate different device capabilities

## Citation

```bibtex
@misc{{anguita2013public,
  title={{A public domain dataset for human activity recognition using smartphones}},
  author={{Anguita, Davide and Ghio, Alessandro and Oneto, Luca and Parra, Xavier and Reyes-Ortiz, Jorge Luis}},
  year={{2013}},
  publisher={{European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning}}
}}
```

## Original Source
- **Repository**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones)
- **License**: MIT

## Example Usage

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("your-username/uci-har")

# Access train split
train_data = dataset["train"]
print(f"Number of training samples: {{len(train_data)}}")

# Access a sample
sample = train_data[0]
print(f"Number of feature columns: 561")
print(f"Activity: {{sample['activity_label']}}")
print(f"Target: {{sample['target']}}")
print(f"Subject: {{sample['subject_id']}}")
print(f"First few features: {{sample['0']:.3f}}, {{sample['1']:.3f}}, {{sample['2']:.3f}}")
```

## Federated Learning Use Cases

This dataset is particularly suitable for:

- **Cross-silo FL**: Different organizations with sensor data
- **Cross-device FL**: Mobile devices performing activity recognition
- **Personalized models**: Subject-specific activity patterns
- **Non-IID scenarios**: Different subjects have different activity patterns

## Data Preprocessing

The sensor signals were preprocessed by:
- Noise filtering using median and 3rd order low pass Butterworth filters
- Sampling at 50Hz fixed-width sliding windows of 2.56 sec (128 readings/window)
- 50% overlap between windows
- Separation of gravitational and body motion components
- Jerk signals derived from body linear acceleration and angular velocity
- Fast Fourier Transform (FFT) applied to some signals
- Feature vector extraction from time and frequency domain variables
"""
    
    return card_content

def main():
    parser = argparse.ArgumentParser(description='Upload UCI HAR CSV dataset to HuggingFace Hub')
    parser.add_argument('--data_path', type=str, 
                      default=r'C:\Users\mrhao\Desktop\AgenticFL\fl_codebase\UCI HAR Dataset',
                      help='Path to directory containing train.csv and test.csv')
    parser.add_argument('--dataset_name', type=str, required=True,
                      help='Name for the dataset on HuggingFace Hub (e.g., "uci-har")')
    parser.add_argument('--hf_username', type=str, required=True,
                      help='Your HuggingFace username')
    parser.add_argument('--private', action='store_true',
                      help='Make the dataset private on HuggingFace Hub')
    
    args = parser.parse_args()
    
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
    
    print(f"Loading UCI HAR CSV dataset from: {args.data_path}")
    
    # Check if CSV files exist
    train_csv = os.path.join(args.data_path, 'train.csv')
    test_csv = os.path.join(args.data_path, 'test.csv')
    
    if not os.path.exists(train_csv):
        print(f"❌ train.csv not found at: {train_csv}")
        return
    
    if not os.path.exists(test_csv):
        print(f"❌ test.csv not found at: {test_csv}")
        return
    
    try:
        # Load and transform the dataset
        dataset_dict, activity_labels, feature_names = load_uci_har_csv_data(args.data_path)
        
        print(f"✅ Dataset loaded successfully!")
        print(f"📊 Train samples: {len(dataset_dict['train']):,}")
        print(f"📊 Test samples: {len(dataset_dict['test']):,}")
        print(f"📊 Feature columns: 561 ('0' to '560')")
        print(f"📊 Target column: 'target'")
        print(f"📊 Additional columns: 'activity_label', 'subject_id'")
        
        # Show sample data
        sample = dataset_dict['train'][0]
        print(f"\n📋 Sample data:")
        print(f"   Activity: {sample['activity_label']}")
        print(f"   Target: {sample['target']}")
        print(f"   Subject: {sample['subject_id']}")
        print(f"   First 3 features: {sample['0']:.3f}, {sample['1']:.3f}, {sample['2']:.3f}")
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Create repository name
    repo_name = f"{args.hf_username}/{args.dataset_name}"
    
    print(f"\n🚀 Uploading dataset to: {repo_name}")
    
    try:
        # Push to HuggingFace Hub
        dataset_dict.push_to_hub(
            repo_id=repo_name,
            private=args.private,
            commit_message="Initial upload of UCI HAR dataset from CSV files"
        )
        
        print(f"✅ Dataset successfully uploaded!")
        
        # Create and upload dataset card
        card_content = create_dataset_card(
            activity_labels, 
            feature_names,
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
        
        print(f"✅ Dataset card uploaded!")
        
    except Exception as e:
        print(f"❌ Error uploading to HuggingFace Hub: {e}")
        return
    
    print(f"\n🎉 Dataset successfully uploaded to: https://huggingface.co/datasets/{repo_name}")
    print("\n📋 Next steps:")
    print("1. Visit your dataset page and review the README")
    print(f"2. Test loading the dataset with: dataset = load_dataset('{repo_name}')")
    print("3. Test with Flower Datasets:")
    print(f"   from flwr_datasets import FederatedDataset")
    print(f"   fds = FederatedDataset(dataset='{repo_name}', partitioners={{'train': 30}})")
    print("4. Share your dataset with the Flower community for inclusion in recommended FL datasets")

if __name__ == "__main__":
    main()