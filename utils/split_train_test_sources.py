import pandas as pd
from sklearn.model_selection import train_test_split

# Constants
DATASET_NAME = "fakebr"
DATASET_PATH = f"../data/{DATASET_NAME}/"
LLM_VERSION = "gpt-5-nano_2025-11-06_18-00-24"
SPLIT = {
    "train": 0.8,
    "validation": 0.10,
    "test": 0.10,
}
RANDOM_STATE = 42

# Check if SPLIT sums to 1.0
if sum(SPLIT.values()) != 1.0:
    raise ValueError("SPLIT values must sum to 1.0")

# Load current dataset splits
original_train = pd.read_csv(DATASET_PATH + "original/train.csv")
original_test = pd.read_csv(DATASET_PATH + "original/test.csv")

extraction_train = pd.read_csv(
    DATASET_PATH + f"claim_extraction/{LLM_VERSION}/train.csv"
)
extraction_test = pd.read_csv(DATASET_PATH + f"claim_extraction/{LLM_VERSION}/test.csv")

normalization_train = pd.read_csv(
    DATASET_PATH + f"claim_normalization/{LLM_VERSION}/train.csv"
)
normalization_test = pd.read_csv(
    DATASET_PATH + f"claim_normalization/{LLM_VERSION}/test.csv"
)

# Concatenate train and test sets
original_full = pd.concat([original_train, original_test], ignore_index=True)
extraction_full = pd.concat([extraction_train, extraction_test], ignore_index=True)
normalization_full = pd.concat(
    [normalization_train, normalization_test], ignore_index=True
)

ids_missing_in_extraction = set(original_full["custom_id"]) - set(
    extraction_full["custom_id"]
)
ids_missing_in_normalization = set(original_full["custom_id"]) - set(
    normalization_full["custom_id"]
)

# Remove samples with missing IDs in extraction or normalization datasets
if ids_missing_in_extraction:
    print(
        f"Removing {len(ids_missing_in_extraction)} samples from original dataset missing in extraction dataset."
    )
    original_full = original_full[
        ~original_full["custom_id"].isin(ids_missing_in_extraction)
    ]

# Remove samples with missing IDs in normalization dataset
if ids_missing_in_normalization:
    print(
        f"Removing {len(ids_missing_in_normalization)} samples from original dataset missing in normalization dataset."
    )
    original_full = original_full[
        ~original_full["custom_id"].isin(ids_missing_in_normalization)
    ]

# Resplit original dataset with scikit-learn
ids = original_full["custom_id"]
labels = original_full["classificacao"]

train_ids, temp_ids = train_test_split(
    ids, test_size=1 - SPLIT["train"], stratify=labels, random_state=RANDOM_STATE
)
temp_labels = labels[ids.isin(temp_ids)]
validation_ids, test_ids = train_test_split(
    temp_ids,
    test_size=SPLIT["test"] / (SPLIT["validation"] + SPLIT["test"]),
    stratify=temp_labels,
    random_state=RANDOM_STATE,
)

# Create new splits for all datasets
new_original_train = original_full[original_full["custom_id"].isin(train_ids)]
new_original_validation = original_full[original_full["custom_id"].isin(validation_ids)]
new_original_test = original_full[original_full["custom_id"].isin(test_ids)]

new_extraction_train = extraction_full[extraction_full["custom_id"].isin(train_ids)]
new_extraction_validation = extraction_full[
    extraction_full["custom_id"].isin(validation_ids)
]
new_extraction_test = extraction_full[extraction_full["custom_id"].isin(test_ids)]

new_normalization_train = normalization_full[
    normalization_full["custom_id"].isin(train_ids)
]
new_normalization_validation = normalization_full[
    normalization_full["custom_id"].isin(validation_ids)
]
new_normalization_test = normalization_full[
    normalization_full["custom_id"].isin(test_ids)
]

# Ensure all datasets have the same IDs in new splits
split_map = {
    "train": (
        new_original_train,
        new_extraction_train,
        new_normalization_train,
        train_ids,
    ),
    "validation": (
        new_original_validation,
        new_extraction_validation,
        new_normalization_validation,
        validation_ids,
    ),
    "test": (new_original_test, new_extraction_test, new_normalization_test, test_ids),
}

for name, (d_orig, d_ext, d_norm, ids) in split_map.items():
    s_orig = set(d_orig["custom_id"])
    s_ext = set(d_ext["custom_id"])
    s_norm = set(d_norm["custom_id"])
    s_expected = set(ids)

    if not (s_orig == s_ext == s_norm == s_expected):
        # build helpful message
        missing_in_ext = s_expected - s_ext
        missing_in_norm = s_expected - s_norm
        extra_in_ext = s_ext - s_expected
        extra_in_norm = s_norm - s_expected

        raise AssertionError(
            f"ID mismatch in {name} split:\n"
            f"  expected {len(s_expected)} ids, orig={len(s_orig)}, ext={len(s_ext)}, norm={len(s_norm)}\n"
            f"  missing_in_ext={sorted(missing_in_ext)[:10]} (showing up to 10)\n"
            f"  missing_in_norm={sorted(missing_in_norm)[:10]}\n"
            f"  extra_in_ext={sorted(extra_in_ext)[:10]}\n"
            f"  extra_in_norm={sorted(extra_in_norm)[:10]}"
        )

# Calculate and print new split sizes and class distribution
for split_name, split_df in zip(
    ["train", "validation", "test"],
    [
        new_original_train,
        new_original_validation,
        new_original_test,
    ],
):
    total_samples = len(split_df)
    class_distribution = split_df["classificacao"].value_counts(normalize=True) * 100
    print(f"{split_name.capitalize()} set: {total_samples} samples")
    print("Class distribution (%):")
    print(class_distribution)
    print()


# Recalculate custom_id to be dataset name + split + sequential integers starting from 1
def recalculate_custom_ids(df, split_name):
    df = df.copy()
    df.reset_index(drop=True, inplace=True)
    df["custom_id"] = [f"{DATASET_NAME}_{split_name}_{i+1}" for i in range(len(df))]
    return df


new_original_train = recalculate_custom_ids(new_original_train, "train")
new_original_validation = recalculate_custom_ids(new_original_validation, "validation")
new_original_test = recalculate_custom_ids(new_original_test, "test")

new_extraction_train = recalculate_custom_ids(new_extraction_train, "train")
new_extraction_validation = recalculate_custom_ids(
    new_extraction_validation, "validation"
)
new_extraction_test = recalculate_custom_ids(new_extraction_test, "test")

new_normalization_train = recalculate_custom_ids(new_normalization_train, "train")
new_normalization_validation = recalculate_custom_ids(
    new_normalization_validation, "validation"
)
new_normalization_test = recalculate_custom_ids(new_normalization_test, "test")

# Save new splits to CSV files
new_original_train.to_csv(DATASET_PATH + "original/new_train.csv", index=False)
new_original_validation.to_csv(
    DATASET_PATH + "original/new_validation.csv", index=False
)
new_original_test.to_csv(DATASET_PATH + "original/new_test.csv", index=False)

new_extraction_train.to_csv(
    DATASET_PATH + f"claim_extraction/{LLM_VERSION}/new_train.csv", index=False
)
new_extraction_validation.to_csv(
    DATASET_PATH + f"claim_extraction/{LLM_VERSION}/new_validation.csv", index=False
)
new_extraction_test.to_csv(
    DATASET_PATH + f"claim_extraction/{LLM_VERSION}/new_test.csv", index=False
)

new_normalization_train.to_csv(
    DATASET_PATH + f"claim_normalization/{LLM_VERSION}/new_train.csv", index=False
)
new_normalization_validation.to_csv(
    DATASET_PATH + f"claim_normalization/{LLM_VERSION}/new_validation.csv", index=False
)
new_normalization_test.to_csv(
    DATASET_PATH + f"claim_normalization/{LLM_VERSION}/new_test.csv", index=False
)

print("Resplitting completed and new datasets saved.")
