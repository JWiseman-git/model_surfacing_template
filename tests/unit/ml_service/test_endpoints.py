from app.endpoints.endpoints import get_chunks

def test_get_chunks():
    text = "I love apples. And I really like oranges."

    result = get_chunks(text)

    assert result == ["I love apples.", "And I really like oranges."]
