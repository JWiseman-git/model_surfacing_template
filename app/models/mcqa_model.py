import torch
import logging
import time
import mlflow
import mlflow.pytorch
from dataclasses import dataclass
from typing import List, Union, Any, Dict, Optional
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForMultipleChoice

logger = logging.getLogger(__name__)

class MCQAModelError(Exception):
    """Base exception for MCQA model errors."""
    pass


class ValidationError(MCQAModelError):
    """Raised when input validation fails."""
    pass


class PredictionError(MCQAModelError):
    """Raised when prediction fails."""
    pass

@dataclass
class MCQAConfig:
    """Configuration for MCQA model."""
    model_directory: Union[Path, str]
    number_of_choices: int = 4
    max_length: int = 512
    device: Optional[str] = None
    batch_size_limit: int = 32
    enable_warmup: bool = True


class MCQAModel:
    """
    Multiple-choice question answering model using a pre-trained transformer.

    This class provides inference capabilities for fill-in-the-blank style
    multiple choice questions using transformer models.

    Example:
       Example:
            >>> config = MCQAConfig(model_directory="./models/mcqa")
            >>> model = MCQAModel(config)
            >>> result = model.predict_blank(
            ...     "The capital of France is [BLANK].",
            ...     ["London", "Paris", "Berlin", "Madrid"]
            ... )
            >>> print(result["predicted_choice"])
    """
    def __init__(self, config: MCQAConfig) -> None:
        """
        Initialise the MCQA model and tokenizer.

        Args:
            model_directory (str): Path to the pre-trained model directory.

        Raises:
            FileNotFoundError: If model directory doesn't exist.
            MCQAModelError: If model initialization fails.
        """
        self.config = config
        model_path = Path(config.model_directory).expanduser().resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Model directory not found: {model_path}")

        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForMultipleChoice.from_pretrained(model_path)
        self.device = torch.device(
            config.device if config.device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)
        self.model.eval()

        self.max_length = config.max_length
        self.num_choices = config.number_of_choices
        self.set_token_limit()
        self.number_of_choices = config.number_of_choices

    def set_token_limit(self) -> None:
        """
        Configure the maximum token length for model inputs depending on the model used.

        Args:
            max_length (int): Maximum sequence length (tokens) for input encoding.
                (NOTE: Defaults to 512 (standard limit for BERT-based models)).

        Raises:
            ValueError: If max_length is non-positive or exceeds model’s configured maximum.
        """

        model_max_len = getattr(self.tokenizer, "model_max_length", None)
        if model_max_len and self.max_length > model_max_len:
            logger.warning(
                f"Requested max_length {self.max_length} exceeds model's max_length {model_max_len}."
            )
            self.max_length = model_max_len
        else:
            pass

        logger.info(f"Token limit set to {self.max_length}")

    @staticmethod
    def _validate_passage(passage: str) -> None:
        """
        Validate a single passage.

        Args:
            passage: The text to validate.

        Raises:
            ValidationError: If passage is invalid.
        """
        if not isinstance(passage, str):
            raise ValidationError(f"Passage must be a string, got {type(passage)}")

        if not passage or not passage.strip():
            raise ValidationError("Passage cannot be empty")

        if "[BLANK]" not in passage:
            raise ValidationError("Passage must contain [BLANK] placeholder")

    def _validate_choices(self, choices: List[str]) -> None:
        """
        Validate a list of choices.

        Args:
            choices: List of answer choices to validate.

        Raises:
            ValidationError: If choices are invalid.
        """
        if not isinstance(choices, list):
            raise ValidationError(f"Choices must be a list, got {type(choices)}")

        if len(choices) != self.num_choices:
            raise ValidationError(
                f"Expected {self.num_choices} choices, "
                f"got {len(choices)}"
            )

        for idx, choice in enumerate(choices):
            if not isinstance(choice, str):
                raise ValidationError(
                    f"Choice {idx} must be a string, got {type(choice)}"
                )
            if not choice.strip():
                raise ValidationError(f"Choice {idx} cannot be empty")

    def _validate_batch_inputs(
            self,
            passages: List[str],
            choices_list: List[List[str]]
    ) -> None:
        """
        Validate batch prediction inputs.

        Args:
            passages: List of passages to validate.
            choices_list: List of choice lists to validate.

        Raises:
            ValidationError: If inputs are invalid.
        """
        if len(passages) != len(choices_list):
            raise ValidationError(
                f"Number of passages ({len(passages)}) must match "
                f"number of choice lists ({len(choices_list)})"
            )

        if len(passages) > self.config.batch_size_limit:
            raise ValidationError(
                f"Batch size {len(passages)} exceeds limit "
                f"{self.config.batch_size_limit}"
            )

        for idx, (passage, choices) in enumerate(zip(passages, choices_list)):
            try:
                self._validate_passage(passage)
                self._validate_choices(choices)
            except ValidationError as e:
                raise ValidationError(f"Invalid input at index {idx}: {e}") from e

    def _predict_helper(self, candidate_texts: List[List[str]]) -> List[Dict[str, Any]]:
        """
        Internal helper for running inference on multiple choice candidates.

        Args:
            candidate_texts (List[List[str]]):
             -   Each element is a list of strings representing answer options
             -   for a single passage with [BLANK] replaced.

        Returns:
            List[Dict[str, Any]]: One result per passage with predicted choice & confidence.

        Raises:
            PredictionError: If prediction fails.
        """
        batch_size = len(candidate_texts)
        num_choices = len(candidate_texts[0])

        start_time = time.time()

        flat_candidates = [cand for choices in candidate_texts for cand in choices]

        encoding = self.tokenizer(
            flat_candidates,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        for key in encoding:
            encoding[key] = encoding[key].view(batch_size, num_choices, -1).to(self.device)

        try:
            with torch.inference_mode():
                outputs = self.model(**encoding)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)
        except torch.cuda.OutOfMemoryError as e:
            raise PredictionError(
                f"GPU out of memory. Try reducing batch size (current: {batch_size})"
            ) from e
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise PredictionError(f"Prediction failed: {e}") from e

        latency_ms = (time.time() - start_time) * 1000
        mlflow.log_metric("mcqa/latency_ms", latency_ms)

        results = []
        for i in range(batch_size):
            pred_idx = preds[i].item()
            best_choice = candidate_texts[i][pred_idx]
            confidence = probs[i][pred_idx].item()
            results.append({
                "predicted_choice": best_choice,
                "confidence": confidence
            })
            mlflow.log_metric(f"mcqa/confidence_{i}", confidence)

        return results

    def predict_blank(self, passage: str, choices: List[str]) -> Dict:
        """
        Predict the correct choice for a passage with a blank.

        Args:
            passage (str): The text containing a [BLANK] placeholder.
            choices (list[str]): A list of exactly 4 possible choices.

        Returns:
            dict: Each item contains:
            - "predicted_choice": str
            - "confidence": float

        Raises:
            ValueError: If the number of choices is not exactly 4.
        """
        self._validate_choices(choices)
        self._validate_passage(passage)

        candidate_texts = [[passage.replace("[BLANK]", choice) for choice in choices]]

        return self._predict_helper(candidate_texts)[0]

    def predict_batch(self, passages: List[str], choices_list: List[List[str]]) -> List[Dict]:
        """
        Predict correct choices for multiple passages with blanks.

        Args:
            passages (List[str]): List of passages containing a [BLANK] placeholder.
            choices_list (List[List[str]]): List of 4-choice lists, one per passage.

        Returns:
            dict: Each item contains:
            - "predicted_choice": str
            - "confidence": float

        Raises:
            ValueError: If lengths mismatch or number of choices != 4 per passage.
        """

        self._validate_batch_inputs(passages, choices_list)

        logger.info(f"Predicting for batch of size: {len(passages)}")

        candidate_texts = [
            [passage.replace("[BLANK]", choice) for choice in choices]
            for passage, choices in zip(passages, choices_list)
        ]

        return self._predict_helper(candidate_texts)
