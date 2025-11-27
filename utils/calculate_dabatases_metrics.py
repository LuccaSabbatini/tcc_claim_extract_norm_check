import pandas as pd

DATASET_NAME = "fakebr"
DATASET_TASK = "claim_normalization"
DATASET_VERSION = "gpt-5-nano_2025-11-06_18-00-24"

train_df = pd.read_csv(
    f"../data/{DATASET_NAME}/{DATASET_TASK}/{DATASET_VERSION}/train.csv"
)
validation_df = pd.read_csv(
    f"../data/{DATASET_NAME}/{DATASET_TASK}/{DATASET_VERSION}/validation.csv"
)
test_df = pd.read_csv(
    f"../data/{DATASET_NAME}/{DATASET_TASK}/{DATASET_VERSION}/test.csv"
)

total_df = pd.concat([train_df, validation_df, test_df], ignore_index=True)

# Calculate average text length
total_df["text_length"] = total_df["text"].apply(len)
average_length = total_df["text_length"].mean()
print(f"Average text length: {average_length:.2f} characters")

# Calculate average number of words per text
total_df["word_count"] = total_df["text"].apply(lambda x: len(x.split()))
average_words = total_df["word_count"].mean()
print(f"Average number of words per text: {average_words:.2f} words")

# Calculate total number of samples
total_samples = len(total_df)
print(f"Total number of samples: {total_samples}")

# Calculate class distribution
class_distribution = total_df["label"].value_counts(normalize=True) * 100
print("Class distribution (%):")
for label, percentage in class_distribution.items():
    print(f"  {label}: {percentage:.2f}%")
