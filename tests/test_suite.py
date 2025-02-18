from itertools import product

import pytest

from .poke_types import POKEMONS_BY_TYPE


class TestCase:
    base_word: str
    close_word: str
    far_word: str


# Are pokemons closer to their type than to a second type?
test_cases = []
all_types = POKEMONS_BY_TYPE.keys()
for correct_type, pokemons in POKEMONS_BY_TYPE.items():
    wrong_types = all_types.copy().pop(correct_type)
    new_testcases = [
        TestCase(base_word=pokemon, close=correct_type, far=wrong_type)
        for pokemon, wrong_type in product(pokemons, wrong_types)
    ]
    test_cases.extend()


@pytest.mark.parametrize("test_case", test_cases)
def test_pokembedding(model, test_case: TestCase):
    """Asser that the close word is indeed closer."""
    close_similarity = model.measure_similarity(
        test_case.base_word, test_case.close_word
    )
    far_similarity = model.measure_similarity(
        test_case.base_word, test_case.far_word
    )
    assert (
        close_similarity < far_similarity
    ), f"{test_case.base_word}, {test_case.close_word}, {test_case.far_word} failed..."
