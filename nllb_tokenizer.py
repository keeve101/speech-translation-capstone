"""
Module name: lang_tokenizers

This module contains the LangTokenizers class, which is used for managing tokenizers for different languages.

Classes:
    LangTokenizers: This class is used for managing tokenizers for different languages.
    - __init__(self, available_langs:list, model:str="facebook/nllb-200-distilled-600M"): This is the constructor for the LangTokenizers class. It takes a required parameter `available_langs`, which is a list of languages for which tokenizers are to be initialized, and an optional parameter `model`, which specifies the NLLB model to be used for creating the tokenizers. The default value is "facebook/nllb-200-distilled-600M".
    - _initialize_tokenizers(self, model:str): This method initializes the tokenizers for the languages specified in the `available_langs` parameter of the constructor.
    - get_tokenizer(self, lang:str): This method returns the tokenizer for the specified language.
    - decode(self, tokenizer, translated_tokens): This method decodes the translated tokens using the specified tokenizer.

"""
from transformers import NllbTokenizerFast


lang_list = {
    "en": "eng_Latn",
    "zh": "zho_Hans",
}


class NllbTokenizer:
    def __init__(
            self, model: str = "facebook/nllb-200-distilled-600M", cache_dir: str|None = None
    ):
        """
        Initializes the LangTokenizers object.

        Parameters:
            model (str): This is an optional parameter that specifies the NLLB model to be used for creating the tokenizers. The default value is "facebook/nllb-200-distilled-600M".
        """
        self._initialize_tokenizers(model, cache_dir)

    def _initialize_tokenizers(self, model: str, cache_dir: str|None):
        """
        Initializes the tokenizers for the languages specified in the `available_langs` parameter of the constructor.

        Parameters:
            model (str): The NLLB model to be used for creating the tokenizers.
        """
        self.tokenizer = {}
        for lang, lang_id in lang_list.items():
            self.tokenizer[lang] = NllbTokenizerFast.from_pretrained(model, use_auth_token=False,
                                                                     src_lang=lang_id, cache_dir=cache_dir)

    def get_lang_id(self, lang: str):
        return lang_list[lang]

    def get_tokenizer(self, lang: str):
        """
        Returns the tokenizer for the specified language.

        Parameters:
            lang (str): The language for which the tokenizer is to be returned.

        Returns:
            type: The tokenizer for the specified language.
        """
        assert (
            lang in self.tokenizer
        ), "INVALID_LANG: {} not in tokenizers list".format(lang)
        return self.tokenizer[lang]

