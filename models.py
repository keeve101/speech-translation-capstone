import whisperx.types
import whisperx
import numpy as np
import torch

from transformers import (
    Wav2Vec2ForCTC,
    AutoProcessor,
    MBartForConditionalGeneration,
    MBart50TokenizerFast,
)
from typing import Union
from config import (
    STORAGE_DIR_MODEL_WHISPER,
    STORAGE_DIR_MODEL_MMS_1B_ALL,
    STORAGE_DIR_MODEL_MBART_LARGE_50_MANY_TO_ONE,
    STORAGE_DIR_MODEL_MBART_LARGE_50_MANY_TO_MANY,
)
from whisperx.asr import FasterWhisperPipeline
from whisperx.types import TranscriptionResult, AlignedTranscriptionResult


class WhisperX:
    def __init__(
        self,
        whisper_arch: str = "medium",
        device: str = "cpu",
        compute_type: str = "int8",
    ):
        self.whisper_arch = whisper_arch
        self.device = device
        self.compute_type = compute_type
        self.model = None

    def get_model_name(self):
        return f"whisper-{self.whisper_arch}"

    def load_model(self) -> FasterWhisperPipeline:
        """Loads the Whisper archtype model."""
        self.model = whisperx.load_model(
            whisper_arch=self.whisper_arch,
            device=self.device,
            compute_type=self.compute_type,
            download_root=STORAGE_DIR_MODEL_WHISPER,
        )
        return self.model

    def get_model(self) -> FasterWhisperPipeline:
        """Returns the loaded model, or loads it if not already loaded."""
        if self.model is None:
            self.load_model()
        return self.model

    def transcribe(
        self, audio: Union[str, np.ndarray], language: str = ""
    ) -> TranscriptionResult:
        """Transcribes audio using the loaded model, transcribes to English by default."""
        audio = audio.astype(np.float32) if isinstance(audio, np.ndarray) else audio
        model = self.get_model()
        transcription = model.transcribe(
            audio=audio, task="transcribe", language=language
        )

        return transcription

    def align_result(
        self,
        result: TranscriptionResult,
        audio: Union[str, np.ndarray],
        device: str = None,
    ) -> AlignedTranscriptionResult:
        """Aligns transcription results with an phenome model."""
        device = device or self.device  # Use instance device if none provided
        model, metadata = whisperx.load_align_model(
            language_code=result["language"], device=device
        )
        alignment_result = whisperx.align(
            transcript=result["segments"],
            model=model,
            align_model_metadata=metadata,
            audio=audio,
            device=device,
            return_char_alignments=False,
        )

        return alignment_result

    def transcribe_align(
        self, audio: Union[str, np.ndarray], language: str = ""
    ) -> AlignedTranscriptionResult:
        """Transcribes audio using the loaded model, transcribes to detected source language by default, though you can set a specific language."""
        audio = audio.astype(np.float32) if isinstance(audio, np.ndarray) else audio
        model = self.get_model()
        transcription = model.transcribe(
            audio=audio, task="transcribe", language=language
        )
        alignment_result = self.align_result(transcription, audio, self.device)

        return alignment_result


class MMS_1B_All:
    def __init__(self, model_id: str = "facebook/mms-1b-all", device: str = "cpu"):
        self.model_id = model_id
        self.model = None
        self.processor = None
        self.device = torch.device(device)

    def get_model_name(self):
        return self.model_id.split("/")[-1]

    def load_model(self):
        """Loads the processor and model for MMS-1B."""
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, cache_dir=STORAGE_DIR_MODEL_MMS_1B_ALL
        )
        self.model = Wav2Vec2ForCTC.from_pretrained(
            self.model_id, cache_dir=STORAGE_DIR_MODEL_MMS_1B_ALL
        )
        self.model.to(self.device)
        return self.processor, self.model

    def transcribe(self, audio: Union[str, np.ndarray], language: str = "eng"):
        """Transcribes audio using the loaded model, transcribes to English by default."""
        if self.processor is None or self.model is None:
            self.load_model()

        audio = audio.astype(np.float32) if isinstance(audio, np.ndarray) else audio
        self.processor.tokenizer.set_target_lang(language)
        self.model.load_adapter(language)

        inputs = self.processor(audio, sampling_rate=16_000, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs).logits

        ids = torch.argmax(outputs, dim=-1)[0]
        transcription = self.processor.decode(ids)

        return transcription


class MBartLarge50ManyToOne:
    def __init__(
        self,
        model_id: str = "facebook/mbart-large-50-many-to-one-mmt",
        device: str = "cpu",
    ):
        self.model_id = model_id
        self.device = torch.device(device)
        self.model = None
        self.tokenizer = None

    def get_model_name(self):
        """Returns the name of the loaded model."""
        return self.model_id.split("/")[-1]

    def load_model(self):
        self.tokenizer = MBart50TokenizerFast.from_pretrained(
            self.model_id, cache_dir=STORAGE_DIR_MODEL_MBART_LARGE_50_MANY_TO_ONE
        )
        self.model = MBartForConditionalGeneration.from_pretrained(
            self.model_id, cache_dir=STORAGE_DIR_MODEL_MBART_LARGE_50_MANY_TO_ONE
        )
        self.model.to(self.device)
        return self.tokenizer, self.model

    def translate(self, text: str, source_lang: str, target_lang: str = "en_XX") -> str:
        """
        Translates text from the source language to the target language (default: English).

        Args:
            text (str): Input text to translate.
            source_lang (str): Source language code (e.g., 'fr_XX' for French).
            target_lang (str): Target language code (default: 'en_XX' for English), since Many-to-One, only to English.

        Returns:
            str: Translated text.
        """
        if self.tokenizer is None or self.model is None:
            self.load_model()

        # Set the source language
        self.tokenizer.src_lang = source_lang

        # Tokenize input text
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        # Generate translation
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.lang_code_to_id[target_lang],
            )

        # Decode the translation
        translated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated_text


class MBartLarge50ManyToMany:
    def __init__(
        self,
        model_id: str = "facebook/mbart-large-50-many-to-many-mmt",
        device: str = "cpu",
    ):
        self.model_id = model_id
        self.device = torch.device(device)
        self.model = None
        self.tokenizer = None

    def get_model_name(self):
        """Returns the name of the loaded model."""
        return self.model_id.split("/")[-1]

    def load_model(self):
        self.tokenizer = MBart50TokenizerFast.from_pretrained(
            self.model_id, cache_dir=STORAGE_DIR_MODEL_MBART_LARGE_50_MANY_TO_MANY
        )
        self.model = MBartForConditionalGeneration.from_pretrained(
            self.model_id, cache_dir=STORAGE_DIR_MODEL_MBART_LARGE_50_MANY_TO_MANY
        )
        self.model.to(self.device)
        return self.tokenizer, self.model

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Translates text from the source language to the target language.
        Args:
            text (str): Input text to translate.
            source_lang (str): Source language code (e.g., 'en_XX' for English).
            target_lang (str): Target language code (e.g., 'zh_CN' for Simplified Chinese).
        Returns:
            str: Translated text.
        """
        if self.tokenizer is None or self.model is None:
            self.load_model()

        # Set the source language
        self.tokenizer.src_lang = source_lang

        # Tokenize input text
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        # Generate translation
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.lang_code_to_id[target_lang],
            )

        # Decode the translation
        translated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated_text
