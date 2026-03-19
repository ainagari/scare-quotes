"""Evaluate saved LLM test-set predictions against the gold standard.

Reads per-instance predictions from id_and_response.txt files produced by
llm_calls.py and reports F1 scores for binary and 3-class settings.

Usage:
    python complete_test_evaluation.py --model olmo
"""

import json
import os
from argparse import ArgumentParser
from collections import Counter

from sklearn.metrics import f1_score

from llm_calls import load_ids, interpret_gold_label, transform_prediction


def load_predictions(path):
    """Load an id_and_response.txt file into a sorted {id: numeric_label} dict."""
    predictions = {}
    with open(path) as f:
        for line in f:
            idi, pred = line.strip().split("\t")
            predictions[int(idi)] = transform_prediction(pred)
    return dict(sorted(predictions.items()))


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["olmo", "llama70B", "qwen"],
        required=True,
    )
    args = parser.parse_args()

    # --- Load corpus and reference labels ---
    corpus = json.load(open("full_postprocessed_expert_annotations_fixed.json"))
    test_ids = load_ids("test")

    test_data = [x for x in corpus if x["data"]["item_order"] in set(test_ids)]

    test_ref_nb = dict(sorted(
        {item["data"]["item_order"]: interpret_gold_label(item, binary=False) for item in test_data}.items()
    ))
    test_ref_b = dict(sorted(
        {item["data"]["item_order"]: interpret_gold_label(item, binary=True) for item in test_data}.items()
    ))

    # IDs that are purely SQ or NSQ (not both), used for the "pure binary" setting
    ids_pure = [k for k, v in test_ref_nb.items() if v != 2]
    test_ref_b_pure = {k: test_ref_b[k] for k in ids_pure}

    print("Gold label distribution (binary):", Counter(test_ref_b.values()))

    # --- Evaluate each saved result directory for the chosen model ---
    for fn in sorted(os.listdir("test_results/")):
        # Directory names follow the pattern: {model}_{prompt}{number}
        # Use rsplit to correctly handle model names that contain underscores
        model_name, prompt_part = fn.rsplit("_", 1)
        if model_name != args.model:
            continue

        prompt_number = int(prompt_part[len("prompt"):])
        binary = prompt_number > 6  # prompts 1-6 are 3-class; 7-12 are binary

        pred_file = f"test_results/{fn}/id_and_response.txt"
        if not os.path.exists(pred_file):
            print(f"  Skipping {fn}: id_and_response.txt not found")
            continue

        predictions = load_predictions(pred_file)

        print(f"\n--- {fn} ({'binary' if binary else '3-class'}) ---")
        if binary:
            predictions_pure = {k: p for k, p in predictions.items() if k in ids_pure}
            print(f"  F1 (binary, all):  {f1_score(list(test_ref_b.values()), list(predictions.values())):.4f}")
            print(f"  F1 (binary, pure): {f1_score(list(test_ref_b_pure.values()), list(predictions_pure.values())):.4f}")
        else:
            print(f"  F1 (3-class, micro): {f1_score(list(test_ref_nb.values()), list(predictions.values()), average='micro'):.4f}")
