import matplotlib.pyplot as plt
import numpy as np
import json
from models import Nllb200, Small100, MBartLarge50ManyToMany

languages_long = {
        'en': "English",
        'zh': "Mandarin Chinese",
        'id': "Indonesian",
}
models = [
    Nllb200(),
    Small100(),
    MBartLarge50ManyToMany()
]
model_names = [
    'NLLB-200',
    'SMaLL-100',
    'mBART-50'
]
lang_codes = [
    "zh",
    'id',
]
datasets = ['fleurs', 'flores']
categories = [
    'WER (1-error)%',
    'SacreBLEU',
    'chrF++',
]

def load_scores(file_path, lang):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            scores = [
                100*(1-data[lang]["wer"]),  # Invert WER so higher is better
                data[lang]["sacrebleu"]["score"],  # Keep original scale
                data[lang]["chr_f"]["score"],  # Keep original scale
            ]
            return scores
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

for dataset in datasets:
    for lang in lang_codes:
        langs = ['en', lang]
        for i in range(2):
            src_lang = langs[i]
            tgt_lang = langs[i-1]
            # Collect scores for all models
            model_scores = {}
            for model_name, model in zip(model_names, models):
                file_path = f'./predictions-translation/{dataset}/{model.get_model_name()}_en_{lang}_evaluation_results.json'
                scores = load_scores(file_path, languages_long[tgt_lang])
                if scores:
                    model_scores[model_name] = scores
            
            
            plt.figure(figsize=(6, 4))
            plt.subplot(polar=True)


            label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(categories)+1)

            for model_name, scores in model_scores.items():
                plt.plot(label_loc, [*scores, scores[0]], label=model_name, linewidth=2)

            plt.title(f'{dataset.upper()} Dataset: {src_lang}-{tgt_lang} Translation Metrics', size=20)
            lines, labels = plt.thetagrids(np.degrees(label_loc), labels=[*categories, ''], fontsize=12)

            plt.legend(fontsize=14, loc='center left', bbox_to_anchor=(-0.5, 0.5))

            plt.savefig(f'predictions-translation/{dataset}_{src_lang}_{tgt_lang}.png')
