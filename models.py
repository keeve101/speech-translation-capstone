import whisperx.types
import whisperx
import numpy as np

from typing import Union
from config import STORAGE_DIR_MODEL_WHISPER
from whisperx.asr import FasterWhisperPipeline
from whisperx.types import TranscriptionResult, AlignedTranscriptionResult


def load_whisper_x(
        whisper_arch: str = "medium", device: str = "cpu", compute_type: str = "int8"
        ) -> FasterWhisperPipeline:
    model = whisperx.load_model(
            whisper_arch=whisper_arch,
            device=device,
            compute_type=compute_type,
            download_root=STORAGE_DIR_MODEL_WHISPER,
            )
    return model


def align_result(
        result: TranscriptionResult, audio: Union[str, np.ndarray], device: str = "cpu"
        ) -> AlignedTranscriptionResult:
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
