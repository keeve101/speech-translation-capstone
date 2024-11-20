from datasets import load_dataset, concatenate_datasets
from fleurs import _FLEURS_LONG_TO_LANG
from config import STORAGE_DIR_DATA_FLEURS
from models import WhisperX, MMS_1B_All
from pathlib import Path
from whisper.normalizers import BasicTextNormalizer, EnglishTextNormalizer

import numpy as np
import json
import torch
import gc
import glob
import evaluate

"""
To run predictions on batches using different models. 

Predictions are stored, and batches are concatenated at the end.
"""


def detect_chinese(text):
    has_chinese = any(
        "\u4E00" <= char <= "\u9FFF" or "\u3400" <= char <= "\u4DBF" for char in text
    )
    return has_chinese


def add_spaces_in_chinese(predictions):
    for idx, prediction in enumerate(predictions):
        reconstruct = ""
        for i, char in enumerate(prediction):
            if detect_chinese(char):
                if i != len(predictions) - 1:
                    reconstruct += char + " "
                else:
                    reconstruct += char
            else:
                reconstruct += char

        predictions[idx] = reconstruct

    return predictions


languages_long = "English,Mandarin Chinese".split(",")
languages_lang = [_FLEURS_LONG_TO_LANG[long.strip()] for long in languages_long]
languages_iso_649_3 = ["eng", "cmn-script_simplified"]
languages_long_to_iso_649_3 = {
    long: iso_649_3 for long, iso_649_3 in zip(languages_long, languages_iso_649_3)
}
long_to_lang_code_translation = {
    long: lang_code for long, lang_code in zip(languages_long, ["en", "zh"])
}

datasets = []

for lang in languages_lang:
    dataset = load_dataset(
        "google/fleurs",
        name=lang,
        cache_dir=STORAGE_DIR_DATA_FLEURS,
        trust_remote_code=True,
        split="test[:10%]",
    )

    datasets.append(dataset)

combined_dataset = concatenate_datasets(datasets)

# Define batch size
batch_size = 16

# Make folder to store predictions
cwd = Path.cwd()
output_folder = cwd / "predictions-translation"
output_folder.mkdir(exist_ok=True)

# Define models
device = "cuda"
models = [WhisperX(whisper_arch="medium", device=device)]

# Define evaluation metrics
wer_metric = evaluate.load("wer")
bleu_metric = evaluate.load("bleu")
evaluation_metrics = [wer_metric, bleu_metric]

for model in models:
    if model.model is not None:
        # Flush the current model from memory
        if model.device == "cuda":
            torch.cuda.empty_cache()
        del model.model
        gc.collect()

    # Preliminary: Try with only English and Mandarin Chinese. Find all unique ids, and filter out the ones that do not have both English and Chinese transcriptions.
    unique_ids = set(combined_dataset["id"])
    for id in unique_ids:
        d_filtered = combined_dataset.filter(lambda x: x["id"] == id)
        if (
            len(d_filtered) == 2
            and d_filtered[0]["language"] != d_filtered[1]["language"]
        ):

            # Predict for batch
            predictions = []

            for idx in range(len(d_filtered)):
                source_sample = d_filtered[idx]
                target_sample = d_filtered[1 - idx]
                audio = source_sample["audio"]["array"].astype(np.float32)
                task = f"{long_to_lang_code_translation[source_sample['language']]}-{long_to_lang_code_translation[target_sample['language']]}"

                batch_output_file_path = (
                    output_folder / f"{model.get_model_name()}_{id}_{task}.json"
                )

                # To skip already processed batches
                if batch_output_file_path.exists():
                    continue

                transcription = model.transcribe(
                    audio,
                    language=long_to_lang_code_translation[target_sample["language"]],
                )

                # Store predictions along with 'id', 'lang_id', 'language', 'lang_group_id'.
                predictions.append(
                    {
                        "id": source_sample["id"],
                        "lang_id": source_sample["lang_id"],
                        "language": source_sample["language"],
                        "lang_group_id": source_sample["lang_group_id"],
                        "prediction": transcription,
                        "source_ground_truth": source_sample["transcription"],
                        "target_ground_truth": target_sample["transcription"],
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

    # Combine segments for WhisperX transcriptions
    if isinstance(model, WhisperX):
        for prediction in all_predictions:
            prediction["prediction"] = " ".join(
                [
                    segment["text"].strip()
                    for segment in prediction["prediction"]["segments"]
                ]
            )

    with open(
        f"{output_folder}/{model.get_model_name()}.json", "w", encoding="utf-8"
    ) as f:
        json.dump(all_predictions, f, indent=4)

    # evaluation_results = {
    #    "model": model.get_model_name(),
    # }

    ## Evaluate the predictions
    # for language in languages_long:
    #    evaluation_results[language] = {}
    #    for metric in evaluation_metrics:

    #        predictions_lang = [
    #            prediction
    #            for prediction in all_predictions
    #            if prediction["language"] == language
    #        ]
    #        predictions = [prediction["prediction"] for prediction in predictions_lang]
    #        references = [prediction["ground_truth"] for prediction in predictions_lang]

    #        if language == "Mandarin Chinese":
    #            predictions = add_spaces_in_chinese(predictions)

    #        normalizer = (
    #            BasicTextNormalizer()
    #            if language != "English"
    #            else EnglishTextNormalizer()
    #        )

    #        predictions = [normalizer(prediction) for prediction in predictions]
    #        references = [normalizer(reference) for reference in references]

    #        results = metric.compute(predictions=predictions, references=references)

    #        # Store results in the dictionary under the corresponding language and metric
    #        evaluation_results[language][metric.name] = results

    ## Write the results to a JSON file
    # output_file_path = f"{output_folder}/{model.get_model_name()}_evaluation_results.json"
    # with open(output_file_path, "w") as output_file:
    #    json.dump(evaluation_results, output_file, indent=4)
