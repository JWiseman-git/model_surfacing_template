from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_mcqa_predict_single():
    """
    Test the /mcqa/predict endpoint for a single passage with exactly 4 choices.

    This test checks:
    - Status code is 200
    - Response contains a 'prediction' field
    - Predicted answer is one of the provided choices
    - Invalid choice count returns HTTP 400
    """
    passage = "The capital of France is [BLANK]."
    choices = ["Paris", "Berlin", "London", "Rome"]

    response = client.post("/mcqa/predict", json={"passage": passage, "choices": choices})
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert data["prediction"]["predicted_choice"] in choices

    bad_choices = ["Paris", "Berlin", "London"]
    response = client.post("/mcqa/predict", json={"passage": passage, "choices": bad_choices})
    assert response.status_code == 400
    assert response.json()["detail"] == "Exactly 4 choices required"


def test_mcqa_predict_chunk():
    """
    Test the /mcqa/predict_chunk endpoint for multiple passages.

    This test checks:
    - Status code is 200
    - Response contains a 'predictions' field
    - Each predicted answer corresponds to the provided choices
    - Mismatched lengths of passages and choices_list returns HTTP 400
    """
    passages = [
        "The capital of France is [BLANK].",
        "The largest planet in the solar system is [BLANK]."
    ]
    choices_list = [
        ["Paris", "Berlin", "London", "Rome"],
        ["Jupiter", "Mars", "Saturn", "Earth"]
    ]

    response = client.post("/mcqa/predict_chunk", json={"passages": passages, "choices_list": choices_list})
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == len(passages)
    for i, pred in enumerate(data["predictions"]):
        assert pred["predicted_choice"] in choices_list[i]

    bad_choices_list = [
        ["Paris", "Berlin", "London", "Rome"]
    ]
    response = client.post("/mcqa/predict_chunk", json={"passages": passages, "choices_list": bad_choices_list})
    assert response.status_code == 400
    assert response.json()["detail"] == "Number of passages does not match number of choices lists"
