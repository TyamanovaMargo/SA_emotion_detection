"""Speech-to-text transcription using distil-whisper (HuggingFace transformers)."""

import re
from typing import Optional, Dict, List, Any
import numpy as np
import torch

from ..config import WhisperConfig
from ..models.schemas import TranscriptionResult
from ..utils.device import get_optimal_device, clear_gpu_memory


FILLER_WORDS = [
    "um", "uh", "er", "ah", "like", "you know", "i mean", 
    "basically", "actually", "literally", "right", "so", 
    "well", "kind of", "sort of", "hmm", "uhh", "umm"
]


class WhisperTranscriber:
    """Transcribe audio using distil-whisper/distil-large-v3 via HuggingFace transformers."""
    
    def __init__(self, config: Optional[WhisperConfig] = None):
        self.config = config or WhisperConfig()
        self._pipe = None
        self._lang_model = None  # lightweight whisper base for language detection
        
        # Resolve device: "auto" -> "cuda:N", "cuda" -> "cuda:N"
        if self.config.device == "auto":
            self.config.device = get_optimal_device(gpu_index=self.config.gpu_index)
        elif self.config.device == "cuda":
            self.config.device = f"cuda:{self.config.gpu_index}"
    
    @property
    def model(self):
        """Lazy load the distil-whisper pipeline."""
        if self._pipe is None:
            device = self.config.device
            model_id = self.config.model_name
            print(f"Loading Whisper model '{model_id}' on device: {device}")
            
            # Clear GPU cache before loading
            if device.startswith("cuda") and torch.cuda.is_available():
                clear_gpu_memory(device)
            
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
            
            torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32
            
            hf_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
            )
            hf_model.to(device)
            
            processor = AutoProcessor.from_pretrained(model_id)
            
            self._pipe = pipeline(
                "automatic-speech-recognition",
                model=hf_model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                torch_dtype=torch_dtype,
                device=device,
            )
            
            print(f"Whisper model loaded successfully on {device}")
        return self._pipe
    
    def _get_lang_model(self):
        """Lazy load lightweight whisper base model for language detection only.
        
        Always loads on CPU — detect_language uses sparse tensors
        which crash on MPS ('aten::_sparse_coo_tensor_with_dims_and_tensors').
        Language detection is fast enough on CPU.
        """
        if self._lang_model is None:
            import whisper
            print("Loading whisper-base for language detection on cpu")
            self._lang_model = whisper.load_model("base", device="cpu")
        return self._lang_model
    
    def unload_model(self):
        """Unload models and free GPU memory."""
        device = self.config.device
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
        if self._lang_model is not None:
            del self._lang_model
            self._lang_model = None
        if device.startswith("cuda"):
            clear_gpu_memory(device)
        print(f"Whisper models unloaded, GPU memory freed on {device}")
    
    def detect_language(self, audio: np.ndarray, sample_rate: int) -> tuple:
        """
        Detect spoken language from audio using whisper-base language detection.
        Uses first 30s of audio. Works even in skip-transcription mode.
        
        Returns:
            Tuple of (language_code, confidence, language_profile)
            language_profile: 'native_english', 'non_native_english', or 'sea_english'
        """
        import whisper
        
        if sample_rate != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
        
        # Use first 30s for detection (Whisper's mel window)
        audio_30s = audio[:16000 * 30]
        
        # Pad to 30s if shorter
        if len(audio_30s) < 16000 * 30:
            audio_30s = np.pad(audio_30s, (0, 16000 * 30 - len(audio_30s)))
        
        lang_model = self._get_lang_model()
        mel = whisper.log_mel_spectrogram(audio_30s).to(lang_model.device)
        _, probs = lang_model.detect_language(mel)
        
        detected_lang = max(probs, key=probs.get)
        confidence = float(probs[detected_lang])
        
        # SEA languages: Indonesian, Malay, Chinese, Tamil, Tagalog, Thai, Vietnamese, Javanese
        sea_languages = {'id', 'ms', 'zh', 'ta', 'tl', 'th', 'vi', 'jw', 'su'}
        
        # Determine language profile
        # Whisper detects LANGUAGE, not ACCENT — non-native speakers with
        # decent accents score 0.85–0.96. Only >0.97 reliably = native.
        if detected_lang == 'en' and confidence > 0.97:
            language_profile = 'native_english'
        elif detected_lang == 'en':
            language_profile = 'non_native_english'
        elif detected_lang in sea_languages:
            language_profile = 'sea_english'
        else:
            language_profile = 'non_native_english'
        
        return detected_lang, confidence, language_profile

    def transcribe(
        self, 
        audio: np.ndarray, 
        sample_rate: int,
        duration: float
    ) -> TranscriptionResult:
        """
        Transcribe audio to text using distil-whisper.
        
        Args:
            audio: Audio waveform as numpy array
            sample_rate: Sample rate of the audio
            duration: Duration in seconds
            
        Returns:
            TranscriptionResult with text and metadata
        """
        if sample_rate != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
        
        # distil-whisper via transformers pipeline
        generate_kwargs = {}
        if self.config.language:
            generate_kwargs["language"] = self.config.language
        
        result = self.model(
            {"raw": audio, "sampling_rate": 16000},
            return_timestamps=True,
            generate_kwargs=generate_kwargs,
        )
        
        text = result.get("text", "").strip()
        
        # Process chunks into segments
        chunks = result.get("chunks", [])
        segments = []
        for chunk in chunks:
            ts = chunk.get("timestamp", (0, 0))
            segments.append({
                "start": ts[0] if ts[0] is not None else 0,
                "end": ts[1] if ts[1] is not None else 0,
                "text": chunk.get("text", "").strip(),
                "words": [],
            })
        
        language = "en"  # distil-whisper is English-only
        
        word_count = len(text.split())
        filler_counts = self._count_filler_words(text)
        total_fillers = sum(filler_counts.values())
        filler_rate = (total_fillers / duration) * 60 if duration > 0 else 0
        
        return TranscriptionResult(
            text=text,
            segments=segments,
            language=language,
            word_count=word_count,
            duration_seconds=duration,
            filler_words=filler_counts,
            filler_word_rate=round(filler_rate, 2)
        )
    
    def _count_filler_words(self, text: str) -> Dict[str, int]:
        """Count filler words in the transcript."""
        text_lower = text.lower()
        counts = {}
        
        for filler in FILLER_WORDS:
            pattern = r'\b' + re.escape(filler) + r'\b'
            matches = re.findall(pattern, text_lower)
            if matches:
                counts[filler] = len(matches)
        
        return counts
