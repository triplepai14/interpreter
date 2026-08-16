"""Offline translation module (Sugoi V4 ja->en, NLLB-200 en->ja)."""

import os
import sys

# Suppress HuggingFace Hub warnings (must be set before import)
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"

from difflib import SequenceMatcher
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.utils import LocalEntryNotFoundError

from . import log
from .models import ModelLoadError

logger = log.get_logger()

# Translation models per direction. Sugoi V4 is a dedicated ja->en model;
# NLLB-200 (distilled 600M) handles en->ja with language tokens.
MODEL_SPECS = {
    "ja-en": {
        "repo_id": "entai2965/sugoi-v4-ja-en-ctranslate2",
        "required_files": ["model.bin", "spm/spm.ja.nopretok.model"],
        "tokenizer_file": "spm/spm.ja.nopretok.model",
        "source_token": None,
        "target_token": None,
        "compute_type": None,  # model is already quantized appropriately
    },
    "en-ja": {
        "repo_id": "entai2965/nllb-200-distilled-600M-ctranslate2",
        "required_files": ["model.bin", "sentencepiece.bpe.model"],
        "tokenizer_file": "sentencepiece.bpe.model",
        "source_token": "eng_Latn",
        "target_token": "jpn_Jpan",
        "compute_type": "int8",  # model ships float32 (2.4GB); int8 keeps RAM/CPU sane
    },
}

# Translation cache defaults
DEFAULT_CACHE_SIZE = 200  # Max cached translations
DEFAULT_SIMILARITY_THRESHOLD = 0.9  # Fuzzy match threshold for cache lookup


def _get_short_path(path: Path) -> str:
    """Convert path to Windows short (8.3) format to handle non-ASCII characters.

    CTranslate2 and SentencePiece (C++ libraries) may fail to load files from paths
    containing non-ASCII characters on Windows. This function converts paths to the
    short 8.3 format which only uses ASCII characters.

    Args:
        path: Path to convert.

    Returns:
        Short path string on Windows if conversion succeeds, otherwise original path string.
    """
    if sys.platform == "win32":
        import ctypes

        buf = ctypes.create_unicode_buffer(512)
        if ctypes.windll.kernel32.GetShortPathNameW(str(path), buf, 512):
            return buf.value
    return str(path)


def _validate_model_files(model_path: Path, required_files: list[str]) -> bool:
    """Check if all required model files exist.

    Args:
        model_path: Path to the model directory.
        required_files: Relative paths that must exist.

    Returns:
        True if all required files exist, False otherwise.
    """
    for file_path in required_files:
        if not (model_path / file_path).exists():
            logger.warning("missing model file", file=file_path)
            return False
    return True


def _get_model_path(repo_id: str, required_files: list[str]) -> Path:
    """Get path to a translation model, downloading if needed.

    Downloads from HuggingFace on first use.
    Model is cached in standard HuggingFace cache (~/.cache/huggingface/).

    Args:
        repo_id: HuggingFace repository id.
        required_files: Files that must exist for the model to work.

    Returns:
        Path to the model directory.

    Raises:
        ModelLoadError: If model files are missing or corrupted.
    """
    # First try to load from cache (no network request)
    try:
        model_path = snapshot_download(
            repo_id=repo_id,
            local_files_only=True,
        )
        model_path = Path(model_path)
        # Validate that required files exist
        if _validate_model_files(model_path, required_files):
            return model_path
        # Files missing - raise error (UI will show "Fix Models" button)
        raise ModelLoadError(
            "Translation model cache is corrupted. Click 'Fix Models' to repair."
        )
    except LocalEntryNotFoundError:
        pass

    # Not cached, download from HuggingFace
    logger.info("downloading translation model", repo=repo_id)
    model_path = Path(snapshot_download(repo_id=repo_id))

    # Validate download completed successfully
    if not _validate_model_files(model_path, required_files):
        raise ModelLoadError(
            "Translation model download incomplete. Click 'Fix Models' to retry."
        )

    return model_path


def _normalize_output(result: str) -> str:
    """Normalize Unicode characters to ASCII equivalents (fixes rendering issues)."""
    return (
        result
        # Curly quotes -> straight quotes
        .replace("‘", "'")  # LEFT SINGLE QUOTATION MARK
        .replace("’", "'")  # RIGHT SINGLE QUOTATION MARK
        .replace("“", '"')  # LEFT DOUBLE QUOTATION MARK
        .replace("”", '"')  # RIGHT DOUBLE QUOTATION MARK
        # Dashes
        .replace("–", "-")  # EN DASH
        .replace("—", "--")  # EM DASH
        .replace("−", "-")  # MINUS SIGN
        # Spaces
        .replace(" ", " ")  # NO-BREAK SPACE
        # Ellipsis
        .replace("…", "...")  # HORIZONTAL ELLIPSIS
    )


def text_similarity(a: str, b: str) -> float:
    """Calculate similarity ratio between two strings (0.0 to 1.0)."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


class TranslationCache:
    """LRU cache for translations with fuzzy key matching."""

    def __init__(
        self,
        max_size: int = DEFAULT_CACHE_SIZE,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ):
        """Initialize the cache.

        Args:
            max_size: Maximum number of entries to store.
            similarity_threshold: Minimum similarity ratio for fuzzy match (0.0-1.0).
        """
        self._cache: dict[str, str] = {}
        self._max_size = max_size
        self._similarity_threshold = similarity_threshold

    def get(self, text: str) -> str | None:
        """Get cached translation, using fuzzy matching if exact match not found.

        Args:
            text: Japanese text to look up.

        Returns:
            Cached translation if found, None otherwise.
        """
        # Try exact match first
        if text in self._cache:
            return self._cache[text]

        # Try fuzzy match
        for cached_text, translation in self._cache.items():
            if text_similarity(text, cached_text) >= self._similarity_threshold:
                return translation

        return None

    def put(self, text: str, translation: str) -> None:
        """Store a translation in the cache.

        Args:
            text: Japanese source text.
            translation: English translation.
        """
        # Simple LRU: remove oldest entry if at capacity
        if len(self._cache) >= self._max_size and text not in self._cache:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        self._cache[text] = translation


class Translator:
    """Offline translator (CTranslate2).

    Directions: "ja-en" (Sugoi V4) and "en-ja" (NLLB-200 distilled 600M).
    """

    def __init__(
        self,
        direction: str = "ja-en",
        cache_size: int = DEFAULT_CACHE_SIZE,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ):
        """Initialize the translator (lazy loading).

        Args:
            direction: Translation direction, one of MODEL_SPECS keys.
            cache_size: Maximum number of translations to cache.
            similarity_threshold: Minimum similarity for fuzzy cache match (0.0-1.0).
        """
        if direction not in MODEL_SPECS:
            raise ValueError(f"unknown translation direction: {direction}")
        self.direction = direction
        self._spec = MODEL_SPECS[direction]
        self._model_path = None
        self._translator = None
        self._tokenizer = None
        self._cache = TranslationCache(cache_size, similarity_threshold)

    def load(self) -> None:
        """Load the translation model, downloading if needed.

        Raises:
            ModelLoadError: If model fails to load.
        """
        if self._translator is not None:
            return

        logger.info("loading translation model", direction=self.direction)

        import ctranslate2
        import sentencepiece as spm

        # Get model path (downloads from HuggingFace if needed)
        self._model_path = _get_model_path(self._spec["repo_id"], self._spec["required_files"])

        load_kwargs = {}
        if self._spec["compute_type"]:
            load_kwargs["compute_type"] = self._spec["compute_type"]

        # Load CTranslate2 model with GPU if available, fallback to CPU
        device = "cpu"
        try:
            cuda_types = ctranslate2.get_supported_compute_types("cuda")
            if cuda_types:
                # Try to load with GPU
                self._translator = ctranslate2.Translator(
                    _get_short_path(self._model_path),
                    device="cuda",
                    **load_kwargs,
                )
                # Test inference to verify CUDA actually works
                # (loading may succeed but inference can fail if cuBLAS is missing)
                self._translator.translate_batch([["テスト"]])
                device = "cuda"
        except Exception as e:
            # GPU failed (load or inference), will use CPU below
            logger.debug("CUDA failed, falling back to CPU", error=str(e))

        if device == "cpu":
            self._translator = ctranslate2.Translator(
                _get_short_path(self._model_path),
                device="cpu",
                **load_kwargs,
            )

        # Load SentencePiece tokenizer
        # Read model as bytes to avoid Unicode path issues on Windows
        # (SentencePiece's C++ layer may not handle non-ASCII paths correctly)
        tokenizer_path = self._model_path / self._spec["tokenizer_file"]
        self._tokenizer = spm.SentencePieceProcessor(model_proto=tokenizer_path.read_bytes())

        device_info = "GPU" if device == "cuda" else "CPU"
        logger.info("translation model ready", direction=self.direction, device=device_info)

    def _encode(self, text: str) -> list[str]:
        """Tokenize source text, adding language tokens if the model needs them."""
        pieces = self._tokenizer.EncodeAsPieces(text)
        if self._spec["source_token"]:
            return [self._spec["source_token"], *pieces, "</s>"]
        return pieces

    def _decode(self, tokens: list[str]) -> str:
        """Decode output tokens to text."""
        if self._spec["target_token"]:
            specials = {self._spec["target_token"], self._spec["source_token"], "</s>"}
            tokens = [t for t in tokens if t not in specials]
            return self._tokenizer.DecodePieces(tokens).strip()
        # Sugoi path: join tokens, clean up SentencePiece markers, normalize
        result = "".join(tokens).replace("▁", " ").strip()
        return _normalize_output(result)

    def translate(self, text: str) -> tuple[str, bool]:
        """Translate Japanese text to English.

        Args:
            text: Japanese text to translate.

        Returns:
            Tuple of (translated English text, was_cached).
        """
        return self.translate_many([text])[0]

    def translate_many(self, texts: list[str]) -> list[tuple[str, bool]]:
        """Translate multiple Japanese texts to English in a single model call.

        Cache hits are served individually; all cache misses are translated
        together in one translate_batch call, which is much faster than
        translating each text in a separate call.

        Args:
            texts: Japanese texts to translate.

        Returns:
            List of (translated English text, was_cached), one per input.
        """
        results: list[tuple[str, bool] | None] = [None] * len(texts)
        miss_indices: list[int] = []

        for i, text in enumerate(texts):
            if not text or not text.strip():
                results[i] = ("", False)
                continue
            # Check cache first (includes fuzzy matching)
            cached = self._cache.get(text)
            if cached is not None:
                results[i] = (cached, True)
            else:
                miss_indices.append(i)

        if miss_indices:
            # Ensure model is loaded
            if self._translator is None:
                self.load()

            # Tokenize inputs and translate them all in one batch
            token_batch = [self._encode(texts[i]) for i in miss_indices]
            target_prefix = None
            if self._spec["target_token"]:
                target_prefix = [[self._spec["target_token"]]] * len(token_batch)
            batch_results = self._translator.translate_batch(
                token_batch,
                target_prefix=target_prefix,
                beam_size=5,
                max_decoding_length=256,
            )

            for i, batch_result in zip(miss_indices, batch_results):
                result = self._decode(batch_result.hypotheses[0])

                # Store in cache
                self._cache.put(texts[i], result)
                results[i] = (result, False)

        return results

    def is_loaded(self) -> bool:
        """Check if the translation model is loaded.

        Returns:
            True if model is loaded, False otherwise.
        """
        return self._translator is not None
