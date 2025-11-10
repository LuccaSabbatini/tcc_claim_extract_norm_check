import os
import pandas as pd

dir_files = os.listdir()

for data_file in dir_files:
    if data_file.endswith(".csv"):
        df = pd.read_csv(data_file)

        new_dir_name = (
            data_file.replace(".csv", "")
            .replace("claim_extraction_", "")
            .replace("claim_normalization_", "")
            .replace("claim_extraction_normalization_", "")
        )

        os.makedirs(new_dir_name, exist_ok=True)

        train_df = df[df["source"] == "train"]
        test_df = df[df["source"] == "test"]

        train_df.to_csv(os.path.join(new_dir_name, "train.csv"), index=False)
        test_df.to_csv(os.path.join(new_dir_name, "test.csv"), index=False)

        print(f"Processed {data_file}:")
        print(f"  Train samples: {len(train_df)}")
        print(f"  Test samples: {len(test_df)}")
