import pytest
import torch
from unittest.mock import MagicMock, patch
from app.models.mcqa_model import MCQAConfig, MCQAModel, ValidationError


@pytest.fixture
def dummy_model_dir(tmp_path):
    d = tmp_path / "dummy_model"
    d.mkdir()
    return d


@pytest.fixture
def mock_tokenizer_and_model():
    with patch("app.models.mcqa_model.AutoTokenizer.from_pretrained") as mock_tok, \
         patch("app.models.mcqa_model.AutoModelForMultipleChoice.from_pretrained") as mock_model:
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": torch.ones((1, 4, 5), dtype=torch.long)}
        mock_tokenizer.model_max_length = 512
        mock_tok.return_value = mock_tokenizer

        mock_model_instance = MagicMock()
        mock_output = MagicMock()
        mock_output.logits = torch.tensor([[0.1, 0.9, 0.0, -0.5]])
        mock_model_instance.return_value = mock_output
        mock_model.return_value = mock_model_instance

        yield mock_tok, mock_model, mock_tokenizer, mock_model_instance


def test_predict_blank_valid(dummy_model_dir, mock_tokenizer_and_model):
    config = MCQAConfig(model_directory=dummy_model_dir)
    model = MCQAModel(config)

    passage = "The capital of France is [BLANK]."
    choices = ["London", "Paris", "Berlin", "Madrid"]

    result = model.predict_blank(passage, choices)

    assert result["predicted_choice"] == "The capital of France is Paris."
    assert isinstance(result["confidence"], float)


def test_predict_blank_invalid_passage(dummy_model_dir, mock_tokenizer_and_model):
    config = MCQAConfig(model_directory=dummy_model_dir)
    model = MCQAModel(config)

    with pytest.raises(ValidationError):
        model.predict_blank("This has no placeholder", ["a", "b", "c", "d"])


def test_predict_batch_too_large(dummy_model_dir, mock_tokenizer_and_model):
    config = MCQAConfig(model_directory=dummy_model_dir, batch_size_limit=1)
    model = MCQAModel(config)

    passages = [
        "The capital of France is [BLANK].",
        "The capital of Germany is [BLANK].",
    ]
    choices_list = [
        ["London", "Paris", "Berlin", "Madrid"],
        ["Rome", "Paris", "Berlin", "Madrid"],
    ]

    with pytest.raises(ValidationError, match="Batch size 2 exceeds limit 1"):
        model.predict_batch(passages, choices_list)

def test_predict_blank_empty_passage(dummy_model_dir, mock_tokenizer_and_model):
    """Edge case: Passage is empty or whitespace only"""
    config = MCQAConfig(model_directory=dummy_model_dir)
    model = MCQAModel(config)

    empty_passages = ["", "   "]
    choices = ["London", "Paris", "Berlin", "Madrid"]

    for passage in empty_passages:
        with pytest.raises(ValidationError, match="Passage cannot be empty"):
            model.predict_blank(passage, choices)


def test_predict_blank_invalid_number_of_choices(dummy_model_dir, mock_tokenizer_and_model):
    """Edge case: Choices list has wrong number of items"""
    config = MCQAConfig(model_directory=dummy_model_dir)
    model = MCQAModel(config)

    passage = "The capital of France is [BLANK]."
    invalid_choices_lists = [
        ["Paris", "London"],
        ["Paris", "London", "Berlin", "Madrid", "Rome"]
    ]

    for choices in invalid_choices_lists:
        with pytest.raises(ValidationError, match="Expected 4 choices"):
            model.predict_blank(passage, choices)

