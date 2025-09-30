import torch
from transformers import AutoTokenizer, AutoModelForMultipleChoice


class MCQAModel:
    def __init__(self, model_dir: str = "./models/mcqa"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForMultipleChoice.from_pretrained(model_dir)
        self.model.eval()

    def predict(self, passage: str, choices: list[str]) -> str:
        # Ensure exactly 4 choices
        if len(choices) != 4:
            raise ValueError("Exactly 4 choices required")

        # Replace [BLANK] with each choice
        texts = [passage.replace("[BLANK]", choice) for choice in choices]

        # Tokenise
        encoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        # Add batch dimension: [batch_size=1, num_choices, seq_len]
        for k in encoding:
            encoding[k] = encoding[k].unsqueeze(0)

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**encoding)

        # Get top choice
        logits = outputs.logits  # shape: [1, num_choices]
        choice_idx = torch.argmax(logits, dim=1).item()
        return choices[choice_idx]
