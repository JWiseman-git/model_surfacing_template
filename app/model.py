import torch
import logging
import time
import mlflow
import mlflow.pytorch

from typing import List, Union, Any, Dict
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForMultipleChoice


logger = logging.getLogger(__name__)


class MCQAModel:
    """
    Multiple-choice question answering model using a pre-trained transformer.
    """
    def __init__(self, number_of_choices: int = 4, model_directory: Union[Path, str] = Path("./models/mcqa")) -> None:
        """
        Initialise the MCQA model and tokenizer.

        Args:
            model_directory (str): Path to the pre-trained model directory.
        """
        model_path = Path(model_directory).expanduser().resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Model directory not found: {model_path}")

        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForMultipleChoice.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self._log_model_params()
        self.set_token_limit()
        self.number_of_choices = number_of_choices

    def _log_model_params(self):
        """Log model parameters to MLflow.
        Parameters logged:
            - model_name (str): The class name of the model (e.g., 'BertForMultipleChoice').
            - tokenizer_name (str): The class name of the tokenizer (e.g., 'BertTokenizer').
            - vocab_size (int): The size of the tokenizer's vocabulary.

        Notes:
            - An active MLflow run must exist when calling this method; otherwise,
            mlflow.log_param will raise an error."""

        mlflow.log_param("model_name", self.model.__class__.__name__)
        mlflow.log_param("tokenizer_name", self.tokenizer.__class__.__name__)
        mlflow.log_param("vocab_size", self.tokenizer.vocab_size)

    def log_model(self, artifact_path="mcqa_model"):
        """
        Save and log the model to MLflow.
        """
        mlflow.pytorch.log_model(
            pytorch_model=self.model,
            artifact_path=artifact_path,
            registered_model_name="MCQAModel"
        )
        logger.info(f"Model logged to MLflow under artifact path: {artifact_path}")

    def set_token_limit(self, max_length: int = 512) -> None:
        """
        Configure the maximum token length for model inputs.

        Args:
            max_length (int): Maximum sequence length (tokens) for input encoding.
                              Defaults to 512 (standard limit for BERT-based models).

        Raises:
            ValueError: If max_length is non-positive or exceeds model’s configured maximum.
        """

        model_max_len = getattr(self.tokenizer, "model_max_length", None)
        if model_max_len and max_length > model_max_len:
            logger.warning(
                f"Requested max_length {max_length} exceeds model's max_length {model_max_len}."
            )
            self.max_length = model_max_len
        else:
            self.max_length = max_length

        logger.info(f"Token limit set to {self.max_length}")

    def _predict_helper(self, candidate_texts: List[List[str]]) -> List[Dict[str, Any]]:
        """
        Internal helper for running inference on multiple choice candidates.

        Args:
            candidate_texts (List[List[str]]):
             -   Each element is a list of strings representing answer options
             -   for a single passage with [BLANK] replaced.

        Returns:
            List[Dict[str, Any]]: One result per passage with predicted choice & confidence.
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
            with torch.no_grad():
                outputs = self.model(**encoding)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise RuntimeError("Prediction failed") from e

        latency_ms = (time.time() - start_time) * 1000
        mlflow.log_metric("latency_ms", latency_ms)

        results = []
        for i in range(batch_size):
            best_choice = candidate_texts[i][preds[i].item()]
            confidence = probs[i][preds[i]].item()
            results.append({
                "predicted_choice": best_choice,
                "confidence": confidence,
                "latency_ms": latency_ms
            })
            mlflow.log_metric(f"confidence_{i}", confidence)

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
        if len(choices) != self.number_of_choices:
            logger.error(f"Number of choices is not exactly {self.number_of_choices}: {len(choices)}")
            raise ValueError(f"Exactly {self.number_of_choices} choices are required.")

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
        if len(passages) != len(choices_list):
            logger.error("Number of passages does not match number of choices lists.")
            raise ValueError("passages and choices lists must have the same length")

        for idx, choices in enumerate(choices_list):
            if len(choices) != self.number_of_choices:
                logger.error(f"Passage {idx} does not have exactly 4 choices: {choices}")
                raise ValueError("Each passage must have exactly 4 choices")

        logger.info(f"Predicting for batch of size: {len(passages)}")

        candidate_texts = [
            [passage.replace("[BLANK]", choice) for choice in choices]
            for passage, choices in zip(passages, choices_list)
        ]

        return self._predict_helper(candidate_texts)
