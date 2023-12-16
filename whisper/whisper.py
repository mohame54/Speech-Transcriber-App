from typing import Literal, Union, Tuple, Optional, List
from transformers import WhisperFeatureExtractor, WhisperTokenizer
import soxr
import soundfile as sf
import numpy as np
from dataclasses import dataclass


# LOCAL
from .decoding import Inference, GreedyDecoding, BeamSearchDecoding, Hypothesis


@dataclass
class WhisperConfig:
    """
    Configuration class for the WhisperInference module.

    Attributes:
    - encoder_path: Path to the encoder model.
    - decoder_path: Path to the decoder model.
    - model_id: Model identifier, default is "openai/whisper-base" this is the only one supported for now. 
    - transcribption_mode: Language mode, default is "English".
    - decoding: Decoding mode, default is "greedy".
    - beam_size: Beam size for beam search decoding, default is 5.
    - eos_id: End-of-sequence token ID, default is 50257.
    - temperature: Temperature for decoding, default is 1.0.
    - top_p: Top-p sampling parameter, default is 0.98.
    - length_penalty: Length penalty for beam search decoding, default is 1.0.
    """
    encoder_path: str
    decoder_path: str
    model_id: str = "openai/whisper-base"
    transcribption_mode: Literal["English", "Arabic"] = "English"
    decoding: Literal["greedy", "beam"] = "greedy"
    beam_size: int = 5
    eos_id: int = 50257
    temperature: float = 1.0
    top_p: float = 0.98
    length_penalty: float = 1.0


class WhisperInference:
    """
    Inference module for transcribing audio using the Whisper model.

    Attributes:
    - processor: WhisperFeatureExtractor for extracting features from audio.
    - tokenizer: WhisperTokenizer for tokenizing transcriptions.
    - decoding: Decoding strategy based on the selected mode.
    """

    def __init__(
        self,
        config: WhisperConfig
    ):
        """
        Initializes the WhisperInference module.

        Args:
        - config: WhisperConfig object containing model configuration.
        """
        # Initialize feature extractor and tokenizer
        self.processor = WhisperFeatureExtractor.from_pretrained(config.model_id)
        self.tokenizer = WhisperTokenizer.from_pretrained(
            config.model_id,
            language=config.transcribption_mode,
            task="transcribe",
        )
        self.config = config
        self.inference = Inference(
            self.config.encoder_path,
            self.config.decoder_path,
            self.config.transcribption_mode,
        )
        self.set_decoding()

    def set_decoding(self, decoding: Optional[str]= None):
        # Initialize inference and decoding strategy based on the selected mode
        decoding = decoding if decoding is not None else self.config.decoding
        if  decoding == "greedy":
            self.decoding = GreedyDecoding(
                self.inference,
                self.config.eos_id,
                self.config.temperature,
                self.config.top_p
            )
        else:
            self.decoding = BeamSearchDecoding(
                self.inference,
                self.config.eos_id,
                self.config.beam_size,
                self.config.length_penalty,
                self.config.top_p,
                self.config.length_penalty
            )    

    def _extract_feats(self, audio)-> np.ndarray:
        """
        Extracts features from the input audio using the feature extractor.

        Args:
        - audio: Input audio as a numpy array.

        Returns:
        - feats: Extracted log mel spectrogram.
        """
        feats = self.processor(audio, sampling_rate=16_000)['input_features']
        return feats

    def __call__(
        self,
        audio: Union[np.ndarray, str],
        max_len: int = 50,
        return_multiple: bool = False,
        **generation_kwargs,
    )-> Union[Hypothesis, List[Hypothesis]]:
        """
        Transcribes the input audio.

        Args:
        - audio: Input audio as a numpy array or file path.
        - max_len: Maximum length of the transcription.
        - return_multiple: Whether to return multiple transcriptions.
        - **generation_kwargs: Additional decoding parameters.

        Returns:
        - Transcription result.
        """
        if isinstance(audio, str):
            audio, _ = load_wav(audio, tr_rate=16_000)   
        # Modify decoding parameters during the function call
        if len(generation_kwargs):
            for k, v in generation_kwargs.items():
                if hasattr(self.decoding, k):
                    setattr(self.decoding, k, v)
        feats = self._extract_feats(audio)
        return self.decoding(feats, max_len, return_multiple)

    def decode(
        self,
        hyps: Union[List[Hypothesis], List[int]],
        skip_special_tokens: Optional[bool] = True,
    ) -> str:
        """
        Decodes the given hypothesis or list of token IDs.

        Args:
        - hyps: Hypothesis or list of token IDs.
        - skip_special_tokens: Whether to skip special tokens during decoding.

        Returns:
        - Decoded transcription.
        """
        if not isinstance(hyps, list):
            hyps = [hyps]
        hyps = [h.tokens for h in hyps]     
        return self.tokenizer.batch_decode(hyps, skip_special_tokens=skip_special_tokens)

    def set_mode(self, mode: Literal["English", "Arabic"]):
        """
        Sets the transcribption mode.

        Args:
        - mode: Language mode (English or Arabic).
        """
        self.decoding.set_mode(mode)


def load_wav(file_path, tr_rate=16_000) -> Tuple[np.ndarray, int]:
    """
    Loads a WAV file and performs resampling if necessary.

    Args:
    - file_path: Path to the WAV file.
    - tr_rate: Target sampling rate, default is 16,000.

    Returns:
    - audio_array: Loaded and resampled audio as a numpy array.
    - tr_rate: Target sampling rate.
    """
    audio_array, orig_rate = sf.read(file_path)
    audio_array = audio_array.astype(np.float32)
    # Resample to the target sample rate
    if orig_rate != tr_rate:
        audio_array = np.apply_along_axis(
            soxr.resample,
            axis=-1,
            arr=audio_array,
            in_rate=orig_rate,
            out_rate=tr_rate,
            quality="soxr_hq",
        )
    return audio_array, tr_rate
