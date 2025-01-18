from datasets import load_dataset
from models import MBartLarge50ManyToMany
from whisper.normalizers import BasicTextNormalizer, EnglishTextNormalizer
from pathlib import Path
from config import STORAGE_DIR_DATA_FLORES_PLUS

import numpy as np
import json
import torch
import gc
import glob
import evaluate

languages_long = "English,Mandarin Chinese".split(",")
languages_glottocodes = ["stan1293", "beij1234"]
languages_mbart_language_ids = ["en_XX", "zh_CN"]

glottocode_to_long = {
    glottocode: long for glottocode, long in zip(languages_glottocodes, languages_long)
}

glottocode_to_mbart_language_id = {
    glottocode: mbart_language_id
    for glottocode, mbart_language_id in zip(
        languages_glottocodes, languages_mbart_language_ids
    )
}

# Make folder to store predictions
cwd = Path.cwd()

# Define models
device = "cuda"
models = [MBartLarge50ManyToMany(device=device)]

# Define evaluation metrics
wer_metric = evaluate.load("wer")
bleu_metric = evaluate.load("bleu")
evaluation_metrics = [wer_metric, bleu_metric]

dataset = load_dataset(
    "openlanguagedata/flores_plus", cache_dir=STORAGE_DIR_DATA_FLORES_PLUS
)

dataset_filtered = dataset["devtest"].filter(
    lambda x: x["glottocode"] in languages_glottocodes
)

unique_ids = set(dataset_filtered["id"])

for model in models:
    if model.model is not None:
        # Flush the current model from memory
        if model.device == "cuda":
            torch.cuda.empty_cache()
        del model.model
        gc.collect()

    output_folder = cwd / f"predictions-translation-{model.get_model_name()}"
    output_folder.mkdir(exist_ok=True)

    for id in unique_ids:
        d_filtered = dataset_filtered.filter(lambda x: x["id"] == id)

        predictions = []

        for idx in range(len(d_filtered)):
            source_sample = d_filtered[idx]
            target_sample = d_filtered[1 - idx]
            task = f"{source_sample['glottocode']}-{target_sample['glottocode']}"

            batch_output_file_path = (
                output_folder / f"{model.get_model_name()}_{id}_{task}.json"
            )

            # To skip already processed batches
            if batch_output_file_path.exists():
                continue

            translation = model.translate(
                text=source_sample["text"],
                source_lang=glottocode_to_mbart_language_id[
                    source_sample["glottocode"]
                ],
                target_lang=glottocode_to_mbart_language_id[
                    target_sample["glottocode"]
                ],
            )

            predictions.append(
                {
                    "id": id,
                    "language": glottocode_to_long[source_sample["glottocode"]],
                    "prediction": translation,
                    "source_ground_truth": source_sample["text"],
                    "target_ground_truth": target_sample["text"],
                    "task": task,
                }
            )

        with open(batch_output_file_path, "w", encoding="utf-8") as f:
            json.dump(predictions, f, indent=4)

    # Combine the batch outputs into a single file
    batch_output_file_paths = glob.glob(
        str(output_folder / f"{model.get_model_name()}_*.json")
    )

    all_predictions = []
    for file in batch_output_file_paths:
        with open(file, "r", encoding="utf-8") as f:
            batch_predictions = json.load(f)
            all_predictions.extend(batch_predictions)

    with open(
        f"{output_folder}/{model.get_model_name()}.json", "w", encoding="utf-8"
    ) as f:
        json.dump(all_predictions, f, indent=4)
