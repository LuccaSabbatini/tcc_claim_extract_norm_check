import logging
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)

# Load the original dataset
original_data = pd.read_csv("./original.csv")

# Rename Columns
original_data = original_data.rename(
    columns={
        "preprocessed_news": "text",
        "label": "classificacao",
    }
)

# Stratified Train-Test Split
train_data, test_data = train_test_split(
    original_data,
    test_size=0.2,
    random_state=42,
    stratify=original_data["classificacao"],
)

# Log the distribution of classes in train and test sets
class_counts = original_data["classificacao"].value_counts()
class_percents = 100 * class_counts / original_data.shape[0]

logging.info("-------------------------------------------")
logging.info(
    "Original dataset class distribution:\n%s (%s percent)",
    class_counts,
    class_percents,
)
logging.info("-------------------------------------------")
logging.info(
    "Train set class distribution:\n%s (%s percent)",
    train_data["classificacao"].value_counts(),
    100 * train_data["classificacao"].value_counts() / train_data.shape[0],
)
logging.info("-------------------------------------------")
logging.info(
    "Test set class distribution:\n%s (%s percent)",
    test_data["classificacao"].value_counts(),
    100 * test_data["classificacao"].value_counts() / test_data.shape[0],
)

# Save the preprocessed datasets
train_data.to_csv("./train.csv", index=False)
test_data.to_csv("./test.csv", index=False)

logging.info("-------------------------------------------")
logging.info("Preprocessing complete. Train and test datasets saved.")
