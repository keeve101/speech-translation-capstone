from datasets import load_dataset, Dataset
from fleurs import _FLEURS_LONG_TO_LANG
from config import STORAGE_DIR_DATA_FLEURS
from models import Nllb200, Small100, MBartLarge50ManyToMany
from pathlib import Path
from tqdm import tqdm
from whisper.normalizers import BasicTextNormalizer, EnglishTextNormalizer
from pandas import DataFrame

import json
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


def add_spaces_in_chinese(prediction):
    reconstruct = ""
    for i, char in enumerate(prediction):
        if detect_chinese(char):
            if i != len(predictions) - 1:
                reconstruct += char + " "
            else:
                reconstruct += char
        else:
            reconstruct += char
    return reconstruct


def extract_and_convert_to_pd(dataset: Dataset, lang: str) -> DataFrame:
    df = dataset.to_pandas()
    return df.rename(columns={'transcription': lang})

def combine_with_pd(combined: DataFrame, dataset2: Dataset, lang2: str):
    df2 = extract_and_convert_to_pd(dataset2, lang2)

    return combined.merge(
        df2[['id', lang2]], 
        on='id',
        how='inner'
    )

languages_long = ["English","Mandarin Chinese"]
lang_codes = ["en", 'zh']
languages_iso_649_3 = ["eng", "cmn-script_simplified"]
languages_lang = [_FLEURS_LONG_TO_LANG[long.strip()] for long in languages_long]
languages_long_to_iso_649_3 = {
    long: iso_649_3 for long, iso_649_3 in zip(languages_long, languages_iso_649_3)
}
long_to_lang_code_translation = {
    long: lang_code for long, lang_code in zip(languages_long, lang_codes)
}

combined_dataset = None

for i, lang in enumerate(languages_lang):
    dataset = load_dataset(
        "google/fleurs",
        name=lang,
        cache_dir=STORAGE_DIR_DATA_FLEURS,
        trust_remote_code=True,
        split="test+validation+train",
    ).select_columns(['id', 'transcription'])

    if combined_dataset is None:
        combined_dataset = extract_and_convert_to_pd(dataset, lang_codes[i])
    else:
        combined_dataset = combine_with_pd(combined_dataset, dataset, lang_codes[i])

combined_dataset = Dataset.from_pandas(combined_dataset.drop_duplicates(['id']))

# Define batch size
batch_size = 32

# Make folder to store predictions
cwd = Path.cwd()
output_folder = cwd / "predictions-translation" / "fleurs"
output_folder.mkdir(exist_ok=True)

# Define models
device = "cuda"
models = [
    Nllb200(device=device),
    # Nllb200(model_id="facebook/nllb-200-distilled-1.3B", device=device),
    Small100(device=device),
    MBartLarge50ManyToMany(device=device)
]

# Define evaluation metrics
wer_metric = evaluate.load("wer")
bleu_metric = evaluate.load("bleu")
evaluation_metrics = [wer_metric, bleu_metric]

for model in models:
    model.unload()

    for i in tqdm(range(0, len(combined_dataset), batch_size)):
        for src_lang in lang_codes:
            for tgt_lang in lang_codes:
                if src_lang == tgt_lang:
                    continue

                task = f"{src_lang}-{tgt_lang}"
                batch_output_file_path = (
                    output_folder / f"{model.get_model_name()}_{task}_batch_{i}.json"
                )

                # To skip already processed batches
                if batch_output_file_path.exists():
                    continue

                batch = combined_dataset.select(
                    list(range(i, min(i + batch_size, len(combined_dataset))))
                )

                normalizers = [
                    BasicTextNormalizer() if tgt_lang != "en" else EnglishTextNormalizer()
                ]

                if tgt_lang == "zh":
                    normalizers.append(add_spaces_in_chinese)

                # Predict for batch
                predictions = []
                for sample in tqdm(batch, leave=False):
                    src_txt = sample[src_lang]
                    tgt_txt = sample[tgt_lang]

                    translation = model.translate(
                        src_txt,
                        source=src_lang,
                        target=tgt_lang,
                    )

                    for normalizer in normalizers:
                        tgt_txt = normalizer(tgt_txt)
                        translation = normalizer(translation)


                    # Store predictions along with 'id', 'lang_id', 'language', 'lang_group_id'.
                    predictions.append(
                        {
                            "id": sample["id"],
                            "src_lang": src_lang,
                            "tgt_lang": tgt_lang,
                            "source_ground_truth": src_txt,
                            "target_ground_truth": tgt_txt,
                            "prediction": translation,
                            "task": task,
                        }
                    )

                with open(batch_output_file_path, "w", encoding="utf-8") as f:
                    json.dump(predictions, f, indent=4)

    model.unload()

    # Combine the batch outputs into a single file
    batch_output_file_paths = glob.glob(
        str(output_folder / f"{model.get_model_name()}_*batch*.json")
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

    evaluation_results = {
       "model": model.get_model_name(),
    }

    for i, predictions in enumerate(all_predictions):
        if type(predictions) != dict:
            print(i, predictions)

    # Evaluate the predictions
    for language in languages_long:
       evaluation_results[language] = {}
       for metric in evaluation_metrics:
           predictions_lang = [
               prediction for prediction in all_predictions if prediction["tgt_lang"] == long_to_lang_code_translation[language]
           ]
           predictions = [prediction["prediction"] for prediction in predictions_lang]
           references = [prediction["target_ground_truth"] for prediction in predictions_lang]

           results = metric.compute(predictions=predictions, references=references)

           # Store results in the dictionary under the corresponding language and metric
           evaluation_results[language][metric.name] = results

    # Write the results to a JSON file
    output_file_path = f"{output_folder}/{model.get_model_name()}_evaluation_results.json"
    with open(output_file_path, "w") as output_file:
       json.dump(evaluation_results, output_file, indent=4)
