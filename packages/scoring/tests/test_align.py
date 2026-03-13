from scoring import align


def test_align_predictions() -> None:
    predicted = """Test O
Test O
Te B-Person
st I-Person
Test I-Person
Test O

Test O
"""

    expected = """Test O
Test O
Test B-Person
Test I-Person
Test O

Test O
"""
    predicted = align(predicted, expected)
    print(predicted)

    assert predicted == expected


def test_align_predictions_multiline() -> None:
    predicted = """Test O
Test O
T B-Person
e I-Person
s I-Person
t I-Person
Test I-Person
Test O

Test O
"""

    expected = """Test O
Test O
Test B-Person
Test I-Person
Test O

Test O
"""
    predicted = align(predicted, expected)
    print(predicted)

    assert predicted == expected
