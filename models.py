import whisperx.types
import whisperx
import numpy as np
import torch

from transformers import Wav2Vec2ForCTC, AutoProcessor
from typing import Union
from config import STORAGE_DIR_MODEL_WHISPER, STORAGE_DIR_MODEL_MMS_1B_ALL
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
    def __init__(self, model_id: str = "facebook/mms-1b-all"):
        self.model_id = model_id
        self.model = None
        self.processor = None

    def load_model(self):
        """Loads the processor and model for MMS-1B."""
        processor = AutoProcessor.from_pretrained(
            self.model_id, cache_dir=STORAGE_DIR_MODEL_MMS_1B_ALL
        )
        model = Wav2Vec2ForCTC.from_pretrained(
            self.model_id, cache_dir=STORAGE_DIR_MODEL_MMS_1B_ALL
        )
        return processor, model

    def transcribe(self, audio: Union[str, np.ndarray], language: str = "eng"):
        """Transcribes audio using the loaded model, transcribes to English by default."""
        audio = audio.astype(np.float32) if isinstance(audio, np.ndarray) else audio
        self.processor.tokenizer.set_target_lang(language)
        self.model.load_adapter(language)

        inputs = self.processor(audio, sampling_rate=16_000, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs).logits

        ids = torch.argmax(outputs, dim=-1)[0]
        transcription = self.processor.decode(ids)

        return transcription
