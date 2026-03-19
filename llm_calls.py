"""LLM inference script for scare-quote classification.

Runs one or more prompt variants against a chosen model and saves per-instance
predictions and evaluation metrics.

Set the HF_TOKEN environment variable before running models that require it
(e.g. llama70B).

Usage example:
    python llm_calls.py --model olmo --subset dev --prompt_types all
"""

import json
import os
import random
from operator import itemgetter

import torch
from argparse import ArgumentParser
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    LogitsProcessor,
    LogitsProcessorList,
)

import generate_prompts
from utils import get_labels_one_level, high_level, fine_level

random.seed(9)

# Set path to llama70B model if using that model; otherwise, this variable is not used.
llama70b_model_path = ""

# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

def transform_prediction(prediction):
    """Map a raw model token to a numeric label (3 = unrecognised)."""
    token = prediction.strip().lower()
    if token == "s":
        return 1
    elif token == "n":
        return 0
    elif token == "b":
        return 2
    else:
        return 3


def interpret_gold_label(item, binary=False):
    """Return the numeric gold label for an annotated item."""
    high_labels = get_labels_one_level(item, high_level, ignore_unsure_ambiguous=True)
    if len(high_labels) == 2:
        return 1 if binary else 2
    elif high_labels == ("Scare quotes",):
        return 1
    elif high_labels == ("Non-scare quotes",):
        return 0


def load_ids(id_type):
    with open(id_type + "_ids.txt") as f:
        return [int(x.strip()) for x in f]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(transformed_predictions, gold, binary=True):
    """Calculate classification metrics in two settings:

    - 'def0'   : unrecognised predictions (value 3) treated as 0.
    - 'notmiss': unrecognised predictions excluded entirely.
    """
    MISSING = 3

    def0_predictions = [p if p != MISSING else 0 for p in transformed_predictions]

    notmissing_gold, notmissing_predictions = [], []
    number_missing = 0
    for p, g in zip(transformed_predictions, gold):
        if p != MISSING:
            notmissing_gold.append(g)
            notmissing_predictions.append(p)
        else:
            number_missing += 1

    def0_metrics, notmiss_metrics = {}, {}
    for metric in [accuracy_score, f1_score, precision_score, recall_score]:
        kwargs = {} if "accuracy" in metric.__name__ or binary else {"average": "micro"}
        def0_metrics[metric.__name__] = metric(gold, def0_predictions, **kwargs)
        notmiss_metrics[metric.__name__] = metric(
            notmissing_gold, notmissing_predictions, **kwargs
        )

    notmiss_metrics["#missing_instances"] = number_missing
    notmiss_metrics["total_instances"] = len(gold)
    notmiss_metrics["pct_missing_instances"] = number_missing / len(gold) * 100

    return {"def0": def0_metrics, "notmiss": notmiss_metrics}


# ---------------------------------------------------------------------------
# Constrained decoding
# ---------------------------------------------------------------------------

class RestrictToLabelSet(LogitsProcessor):
    """Restrict generation to one of a fixed set of allowed token sequences."""

    def __init__(self, allowed_sequences):
        self.allowed_sequences = allowed_sequences
        self.start_len = 0

    def set_start_len(self, input_ids):
        self.start_len = input_ids.shape[1]

    def __call__(self, input_ids, scores):
        new_mask = torch.full_like(scores, float("-inf"))
        prefix_len = max(0, input_ids.shape[1] - self.start_len)
        for seq in self.allowed_sequences:
            if prefix_len < len(seq):
                if list(input_ids[0][self.start_len:]) == seq[:prefix_len]:
                    new_mask[:, seq[prefix_len]] = scores[:, seq[prefix_len]]
        return new_mask


# ---------------------------------------------------------------------------
# Prompt configuration
# ---------------------------------------------------------------------------

prompt_dict = {
    1:  {"context": False, "binary": False, "examples": 1, "verbosity": 3},
    2:  {"context": False, "binary": False, "examples": 1, "verbosity": 2},
    3:  {"context": False, "binary": False, "examples": 1, "verbosity": 1},
    4:  {"context": False, "binary": False, "examples": 2, "verbosity": 3},
    5:  {"context": False, "binary": False, "examples": 2, "verbosity": 2},
    6:  {"context": False, "binary": False, "examples": 2, "verbosity": 1},
    7:  {"context": False, "binary": True,  "examples": 1, "verbosity": 3},
    8:  {"context": False, "binary": True,  "examples": 1, "verbosity": 2},
    9:  {"context": False, "binary": True,  "examples": 1, "verbosity": 1},
    10: {"context": False, "binary": True,  "examples": 2, "verbosity": 3},
    11: {"context": False, "binary": True,  "examples": 2, "verbosity": 2},
    12: {"context": False, "binary": True,  "examples": 2, "verbosity": 1},
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--prompt_types", help='Prompt type number, "all" or "best": "all" tries all prompts, "best" tries the best prompt for each model based on dev set results (specified below).")')
    parser.add_argument("--model", choices=["olmo", "llama70B", "qwen"])
    parser.add_argument("--subset", choices=["all", "iaa", "dev", "test"])
    args = parser.parse_args()

    # --- Determine which prompt types to run ---
    if args.prompt_types == "all":
        prompt_types = range(1, 13)    
    elif args.prompt_types == "best":
        best_prompts = {"olmo": [11, 2], "llama70B": [12, 4], "qwen": [12, 4]}
        prompt_types = best_prompts[args.model]
    else:
        prompt_types = [int(args.prompt_types)]

    # --- Load model and tokenizer ---
    hf_token = os.environ.get("HF_TOKEN")

    if args.model == "olmo":
        model = AutoModelForCausalLM.from_pretrained("allenai/OLMo-2-1124-7B-Instruct")
        tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-1124-7B-Instruct")
    elif args.model == "qwen":
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")    
    elif args.model == "llama70B":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model_path = llama70b_model_path
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, quantization_config=quant_config, device_map="auto", trust_remote_code=True
        )    

    if args.model != "llama70B":
        model = model.to("cuda")
    model.eval()

    # --- Load dataset and determine ordered instance IDs ---
    corpus_full = json.load(open("annotated_corpus.json"))
    corpus_by_id = {x["data"]["item_order"]: x for x in corpus_full}

    split_ids = {id_type: load_ids(id_type) for id_type in ["dev", "test", "iaa"]}

    if args.subset == "all":
        ordered_ids = split_ids["dev"] + split_ids["test"] + split_ids["iaa"]    
    else:
        ordered_ids = split_ids[args.subset]

    # Preserve the exact order from the ID files
    corpus = [corpus_by_id[i] for i in ordered_ids if i in corpus_by_id]

    # --- Loop over prompt types ---
    f1_by_prompttype = {}

    for prompt_type in prompt_types:
        print(f"Running prompt type {prompt_type}...")
        prompt_characteristics = prompt_dict[prompt_type]
        binary = prompt_characteristics["binary"]

        if binary and args.subset == "iaa":
            continue

        allowed_labels = ["S", "N"] if binary else ["S", "N", "B"]
        allowed_token_ids = [tokenizer.encode(label, add_special_tokens=False) for label in allowed_labels]

        out_dir = f"{args.subset}_results/{args.model}_prompt{prompt_type}/"
        os.makedirs(out_dir, exist_ok=True)

        with open(f"prompt_texts/prompt{prompt_type}.txt") as f:
            prompt_template = f.read().strip()

        predictions = []
        gold = []
        predicted_ids = []

        for item in corpus:
            item_id = item["data"]["item_order"]
            full_prompt = prompt_template.replace(
                "[UTTERANCE HERE]",
                generate_prompts.create_instance_for_prompt(
                    item["data"]["dialogue"], context=prompt_characteristics["context"]
                ),
            )
            gold_label = interpret_gold_label(item, binary=binary)

            message = [{"role": "user", "content": full_prompt}]
            inputs = tokenizer.apply_chat_template(
                message, return_tensors="pt", add_generation_prompt=True, tokenize=True
            )
            inputs = inputs.to("cuda")

            processor = RestrictToLabelSet(allowed_token_ids)
            processor.set_start_len(inputs)
            processors = LogitsProcessorList([processor])

            with torch.no_grad():
                response = model.generate(
                    inputs, max_new_tokens=1, do_sample=False, logits_processor=processors
                )

            readable_output = tokenizer.batch_decode(response, skip_special_tokens=True)[0]

            # Save full model output for debugging
            with open(out_dir + f"{item_id}_response.txt", "w") as out:
                out.write(readable_output)

            last_token = readable_output.split()[-1]
            predictions.append(last_token)
            gold.append(gold_label)
            predicted_ids.append(item_id)

        # Save predictions paired with IDs (tab-separated, in corpus order)
        with open(out_dir + "id_and_response.txt", "w") as out:
            for item_id, pred in zip(predicted_ids, predictions):
                out.write(f"{item_id}\t{pred}\n")

        # Evaluate and save metrics
        transformed_predictions = [transform_prediction(p) for p in predictions]
        try:
            results = evaluate(transformed_predictions, gold, binary=binary)
            print(results)
            f1_by_prompttype[prompt_type] = results["def0"]["f1_score"]
            json.dump(results, open(out_dir + "results.json", "w"))
        except Exception as e:
            print(f"Evaluation error for prompt {prompt_type}: {e}")

    # --- Report best prompt per category ---
    print("Results by prompt type:", f1_by_prompttype)
    binary_results = {k: v for k, v in f1_by_prompttype.items() if prompt_dict[k]["binary"]}
    nonbinary_results = {k: v for k, v in f1_by_prompttype.items() if not prompt_dict[k]["binary"]}
    if binary_results:
        print("Best binary prompt:", max(binary_results.items(), key=itemgetter(1))[0])
    if nonbinary_results:
        print("Best non-binary prompt:", max(nonbinary_results.items(), key=itemgetter(1))[0])
