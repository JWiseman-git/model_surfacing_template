import torch
import logging
import mlflow
import mlflow.pytorch

from pathlib import Path
from transformers import AutoTokenizer, AutoModelForMultipleChoice

logger = logging.getLogger(__name__)


class MCQAModel:
    """
    Multiple-choice question answering model using a pre-trained transformer.
    """
    def __init__(self, model_directory: Path | str = Path("./models/mcqa")) -> None:
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
        self.model.eval()
        self._log_model_params()

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
            registered_model_name="MCQAModel"  # Optional if using Model Registry
        )
        logger.info(f"Model logged to MLflow under artifact path: {artifact_path}")

    def predict_blank(self, passage: str, choices: list[str]) -> str:
        """
        Predict the correct choice for a passage with a blank.

        Args:
            passage (str): The text containing a [BLANK] placeholder.
            choices (list[str]): A list of exactly 4 possible choices.

        Returns:
            str: The predicted correct choice.

        Raises:
            ValueError: If the number of choices is not exactly 4.
        """
        if len(choices) != 4:
            logger.error(f"Number of choices is not exactly 4: {len(choices)}")
            raise ValueError("Exactly 4 choices are required.")

        logger.info("Input passage: %s", passage)
        logger.info("Choices: %s", choices)

        candidate_texts = [passage.replace("[BLANK]", choice) for choice in choices]

        encoding = self.tokenizer(
            candidate_texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        for key in encoding:
            encoding[key] = encoding[key].unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(**encoding)

        logits = outputs.logits
        selected_index = torch.argmax(logits, dim=1).item()

        probs = torch.softmax(logits, dim=1)
        mlflow.log_metric("prediction_confidence", probs[0, selected_index].item())

        return choices[selected_index]
