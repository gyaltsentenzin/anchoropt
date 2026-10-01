#!/usr/bin/env python3
"""
Split input_data.jsonl into train and test sets.

Train: 35 samples per category (140 total)
Test: 15 samples per category (60 total)

Categories: Single-Hop, Multi-Hop, Parallel Single-Hop, Parallel Multi-Hop
"""

import json
import os
from collections import defaultdict
from pathlib import Path

# Configuration
INPUT_FILE = "data/input_data.jsonl"
TRAIN_FILE = "data/input_data_train.jsonl"
TEST_FILE = "data/input_data_test.jsonl"

# Split ratio per category
TRAIN_PER_CATEGORY = 35
TEST_PER_CATEGORY = 15
TOTAL_PER_CATEGORY = 50

# Map data_source to category names for consistency
CATEGORY_MAP = {
    "Single-Hop": "Single-Hop",
    "Multi-Hop": "Multi-Hop",
    "Parallel Single-Hop": "Parallel Single-Hop",
    "Parallel Multi-Hop": "Parallel Multi-Hop",
}


def main():
    # Get script directory
    script_dir = Path(__file__).parent
    input_path = script_dir / INPUT_FILE
    train_path = script_dir / TRAIN_FILE
    test_path = script_dir / TEST_FILE

    # Validate input file exists
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return False

    # Read all data and organize by category
    data_by_category = defaultdict(list)
    total_samples = 0

    print("Reading input data...")
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            sample = json.loads(line)
            category = sample.get("data_source", "Unknown")
            data_by_category[category].append(sample)
            total_samples += 1

    print(f"Total samples read: {total_samples}")
    print("\nCategory distribution:")
    for category, samples in sorted(data_by_category.items()):
        print(f"  {category}: {len(samples)} samples")

    # Validate we have the right number of samples per category
    all_valid = True
    for category, samples in data_by_category.items():
        if len(samples) != TOTAL_PER_CATEGORY:
            print(
                f"Warning: Expected {TOTAL_PER_CATEGORY} samples in '{category}', "
                f"but got {len(samples)}"
            )
            all_valid = False

    if not all_valid:
        print("Warning: Category distribution is not as expected, but continuing...")

    # Split each category
    train_samples = []
    test_samples = []

    print("\nSplitting data...")
    for category, samples in sorted(data_by_category.items()):
        # Take first TRAIN_PER_CATEGORY for training, rest for testing
        train_portion = samples[:TRAIN_PER_CATEGORY]
        test_portion = samples[TRAIN_PER_CATEGORY : TRAIN_PER_CATEGORY + TEST_PER_CATEGORY]

        train_samples.extend(train_portion)
        test_samples.extend(test_portion)

        print(
            f"  {category}: {len(train_portion)} train, {len(test_portion)} test"
        )

    print(f"\nTotal train samples: {len(train_samples)}")
    print(f"Total test samples: {len(test_samples)}")

    # Write train data
    print(f"\nWriting train data to {TRAIN_FILE}...")
    with open(train_path, "w", encoding="utf-8") as f:
        for sample in train_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    # Write test data
    print(f"Writing test data to {TEST_FILE}...")
    with open(test_path, "w", encoding="utf-8") as f:
        for sample in test_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    # Verify files were created
    if train_path.exists() and test_path.exists():
        train_lines = sum(1 for _ in open(train_path, "r"))
        test_lines = sum(1 for _ in open(test_path, "r"))
        print(f"\nSplit completed successfully!")
        print(f"Train file: {train_lines} samples")
        print(f"Test file: {test_lines} samples")
        return True
    else:
        print("Error: Failed to create output files")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
