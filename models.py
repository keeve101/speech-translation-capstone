import whisperx.types
import whisperx
import numpy as np
import torch
import gc

from transformers import Wav2Vec2ForCTC, AutoProcessor, AutoModelForSeq2SeqLM, M2M100ForConditionalGeneration
from typing import Union
from config import *
from small100_tokenization import SMALL100Tokenizer
from nllb_tokenizer import NllbTokenizer
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

    def unload(self):
        if self.model is not None:
            del self.model
        # Flush the current model from memory
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

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

    def unload(self):
        if self.model is not None:
            del self.model
        if self.processor is not None:
            del self.processor
        # Flush the current model from memory
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

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


### TRANSLATION ###

class Nllb200:
    def __init__(self, model_id: str = "facebook/nllb-200-distilled-600M", device: str = "cpu"):
        self.model_id = model_id
        self.tokenizers: NllbTokenizer|None = None
        self.model: AutoModelForSeq2SeqLM|None = None
        self.device = torch.device(device)

    def get_model_name(self):
        return self.model_id.split("/")[-1]

    def load_model(self):
        self.tokenizers = NllbTokenizer(self.model_id, cache_dir=STORAGE_DIR_MODEL_NLLB)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id, cache_dir=STORAGE_DIR_MODEL_NLLB)

        self.model.to(self.device)
        return self.tokenizers, self.model

    def unload(self):
        if self.model is not None:
            del self.model
        if self.tokenizers is not None:
            del self.tokenizers
        # Flush the current model from memory
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    def translate(self, text: str, source: str = "en", target: str = "en", max_length=1000):
        if self.tokenizers is None or self.model is None:
            self.load_model()

        tokenizer = self.tokenizers.get_tokenizer(source)

        with torch.no_grad():
            inputs = tokenizer(text, return_tensors="pt", padding=True).to(self.device)

            translated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids(self.tokenizers.get_lang_id(target)),
                max_length=max_length
            )
            translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
            return translation

class Small100:
    def __init__(self, model_id: str = "alirezamsh/small100", device: str = "cpu"):
        self.model_id = model_id
        self.tokenizer = None
        self.model = None
        self.device = torch.device(device)

    def get_model_name(self):
        return self.model_id.split("/")[-1]

    def load_model(self):
        self.tokenizer = SMALL100Tokenizer.from_pretrained("alirezamsh/small100", cache_dir=STORAGE_DIR_MODEL_SMALL)
        self.model = M2M100ForConditionalGeneration.from_pretrained("alirezamsh/small100", cache_dir=STORAGE_DIR_MODEL_SMALL)

        self.model.to(self.device)
        return self.tokenizer, self.model

    def unload(self):
        if self.model is not None:
            del self.model
        if self.tokenizer is not None:
            del self.tokenizer
        # Flush the current model from memory
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    def translate(self, text: str, source: str = "en", target: str = "en"):
        if self.tokenizer is None or self.model is None:
            self.load_model()

        self.tokenizer.tgt_lang = target
        encoded_text = self.tokenizer(text, return_tensors="pt").to(self.device)
        generated_tokens = self.model.generate(**encoded_text)
        return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
