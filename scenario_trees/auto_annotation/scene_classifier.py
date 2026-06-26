"""CLIP-based scene classifier for autonomous driving scenarios.

Uses OpenAI CLIP (via the open_clip package) to classify driving scenes
by road type, weather conditions, and time of day using zero-shot
text-image similarity.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import open_clip


class CLIPSceneClassifier:
    """Zero-shot scene classifier using CLIP for autonomous driving contexts.

    Classifies driving camera images into categories for road type, weather,
    and time of day using cosine similarity between CLIP image embeddings and
    pre-defined text prompts.

    The model is lazily initialized on first use to avoid loading heavy weights
    at import time.

    Parameters
    ----------
    model_name : str
        The CLIP model architecture name (default: "ViT-B-32").
    pretrained : str
        The pretrained weights identifier (default: "laion2b_s34b_b79k").
    device : str or None
        Device to run inference on. If None, auto-detects CUDA availability.
    """

    # Text prompts for each classification category
    ROAD_TYPE_PROMPTS: Dict[str, str] = {
        "highway": "a photo of a highway",
        "urban": "a photo of an urban street",
        "rural": "a photo of a rural road",
        "intersection": "a photo of an intersection",
    }

    WEATHER_PROMPTS: Dict[str, str] = {
        "clear": "a photo taken in clear weather",
        "rainy": "a photo taken in rainy weather",
        "snowy": "a photo taken in snowy weather",
        "foggy": "a photo taken in foggy weather",
        "overcast": "a photo taken in overcast weather",
    }

    TIME_OF_DAY_PROMPTS: Dict[str, str] = {
        "daylight": "a photo taken during daylight",
        "dawn": "a photo taken at dawn",
        "dusk": "a photo taken at dusk",
        "night": "a photo taken at night",
    }

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: Optional[str] = None,
    ) -> None:
        self._model_name = model_name
        self._pretrained = pretrained
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Lazy-loaded model components
        self._model: Optional[open_clip.CLIP] = None
        self._preprocess = None
        self._tokenizer = None

        # Cached text embeddings per category (computed once per category)
        self._text_embedding_cache: Dict[str, Tuple[List[str], torch.Tensor]] = {}

    def _ensure_model_loaded(self) -> None:
        """Load the CLIP model, preprocessing transform, and tokenizer on first use."""
        if self._model is not None:
            return

        model, _, preprocess = open_clip.create_model_and_transforms(
            self._model_name,
            pretrained=self._pretrained,
            device=self._device,
        )
        model.eval()

        self._model = model
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(self._model_name)

    def _get_text_embeddings(
        self, category: str, prompts: Dict[str, str]
    ) -> Tuple[List[str], torch.Tensor]:
        """Get cached text embeddings for a category, computing them if needed.

        Parameters
        ----------
        category : str
            Cache key for this set of prompts.
        prompts : Dict[str, str]
            Mapping from label name to text prompt string.

        Returns
        -------
        Tuple[List[str], torch.Tensor]
            Ordered list of label names and their corresponding normalized
            text embeddings tensor of shape (num_prompts, embed_dim).
        """
        if category in self._text_embedding_cache:
            return self._text_embedding_cache[category]

        self._ensure_model_loaded()

        labels = list(prompts.keys())
        texts = list(prompts.values())

        tokens = self._tokenizer(texts).to(self._device)

        with torch.no_grad():
            text_features = self._model.encode_text(tokens)
            # L2-normalize for cosine similarity
            text_features = F.normalize(text_features, dim=-1)

        self._text_embedding_cache[category] = (labels, text_features)
        return labels, text_features

    def _encode_image(self, image: np.ndarray) -> torch.Tensor:
        """Encode a numpy image array into a normalized CLIP embedding.

        Parameters
        ----------
        image : np.ndarray
            Input image as HWC uint8 array (BGR or RGB). The image is
            converted to RGB PIL Image before CLIP preprocessing.

        Returns
        -------
        torch.Tensor
            Normalized image embedding of shape (1, embed_dim).
        """
        self._ensure_model_loaded()

        # Convert numpy HWC BGR/RGB uint8 to PIL RGB Image
        # Assume BGR input (common for OpenCV) and convert to RGB
        if image.ndim == 3 and image.shape[2] == 3:
            image_rgb = image[:, :, ::-1].copy()
        else:
            image_rgb = image.copy()

        pil_image = Image.fromarray(image_rgb, mode="RGB")

        # Apply CLIP preprocessing (resize, center crop, normalize)
        image_tensor = self._preprocess(pil_image).unsqueeze(0).to(self._device)

        with torch.no_grad():
            image_features = self._model.encode_image(image_tensor)
            image_features = F.normalize(image_features, dim=-1)

        return image_features

    def _classify(
        self, image: np.ndarray, category: str, prompts: Dict[str, str]
    ) -> Dict[str, float]:
        """Classify an image against a set of text prompts using cosine similarity.

        Parameters
        ----------
        image : np.ndarray
            Input image as HWC uint8 array.
        category : str
            Cache key for the text embeddings.
        prompts : Dict[str, str]
            Mapping from label name to text prompt.

        Returns
        -------
        Dict[str, float]
            Mapping from label name to softmax probability score.
        """
        image_features = self._encode_image(image)
        labels, text_features = self._get_text_embeddings(category, prompts)

        # Compute cosine similarity (features are already normalized)
        # image_features: (1, D), text_features: (N, D)
        similarities = (image_features @ text_features.T).squeeze(0)

        # Apply softmax to convert similarities to probabilities
        # Scale by 100 (CLIP's logit scale) for sharper distributions
        logit_scale = self._model.logit_scale.exp()
        probabilities = F.softmax(similarities * logit_scale, dim=-1)

        # Convert to Python dict with float values
        probs_np = probabilities.cpu().numpy()
        return {label: float(probs_np[i]) for i, label in enumerate(labels)}

    def classify_road_type(self, image: np.ndarray) -> Dict[str, float]:
        """Classify the road type in a driving scene image.

        Categories: highway, urban, rural, intersection.

        Parameters
        ----------
        image : np.ndarray
            Input image as HWC uint8 array (BGR or RGB).

        Returns
        -------
        Dict[str, float]
            Probability scores for each road type category.
            Values sum to 1.0.
        """
        return self._classify(image, "road_type", self.ROAD_TYPE_PROMPTS)

    def classify_weather(self, image: np.ndarray) -> Dict[str, float]:
        """Classify weather conditions in a driving scene image.

        Categories: clear, rainy, snowy, foggy, overcast.

        Parameters
        ----------
        image : np.ndarray
            Input image as HWC uint8 array (BGR or RGB).

        Returns
        -------
        Dict[str, float]
            Probability scores for each weather category.
            Values sum to 1.0.
        """
        return self._classify(image, "weather", self.WEATHER_PROMPTS)

    def classify_time_of_day(self, image: np.ndarray) -> Dict[str, float]:
        """Classify time of day in a driving scene image.

        Categories: daylight, dawn, dusk, night.

        Parameters
        ----------
        image : np.ndarray
            Input image as HWC uint8 array (BGR or RGB).

        Returns
        -------
        Dict[str, float]
            Probability scores for each time-of-day category.
            Values sum to 1.0.
        """
        return self._classify(image, "time_of_day", self.TIME_OF_DAY_PROMPTS)

    def classify_scene(self, image: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Classify a driving scene across all categories.

        Performs road type, weather, and time-of-day classification in a
        single call. The image embedding is computed once and reused across
        all three classifications for efficiency.

        Parameters
        ----------
        image : np.ndarray
            Input image as HWC uint8 array (BGR or RGB).

        Returns
        -------
        Dict[str, Dict[str, float]]
            Nested dictionary with keys "road_type", "weather", and
            "time_of_day", each mapping to their respective probability
            distributions.
        """
        # Encode image once, then classify against each category's text embeddings
        image_features = self._encode_image(image)

        results: Dict[str, Dict[str, float]] = {}

        categories = {
            "road_type": self.ROAD_TYPE_PROMPTS,
            "weather": self.WEATHER_PROMPTS,
            "time_of_day": self.TIME_OF_DAY_PROMPTS,
        }

        logit_scale = self._model.logit_scale.exp()

        for category_name, prompts in categories.items():
            labels, text_features = self._get_text_embeddings(category_name, prompts)

            similarities = (image_features @ text_features.T).squeeze(0)
            probabilities = F.softmax(similarities * logit_scale, dim=-1)

            probs_np = probabilities.cpu().numpy()
            results[category_name] = {
                label: float(probs_np[i]) for i, label in enumerate(labels)
            }

        return results
