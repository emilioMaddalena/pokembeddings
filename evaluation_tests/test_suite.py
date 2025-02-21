from itertools import product

import pytest

from model import Word2Vec

from .poke_types import POKEMONS_BY_TYPE


class WordGroup:  # noqa: D101
    def __init__(self, base_word: str, close_word: str, far_word: str):  # noqa: D107
        self.base_word = base_word
        self.close_word = close_word
        self.far_word = far_word


# Are pokemons closer to their type than to a second type?
test_cases = []
all_types = list(POKEMONS_BY_TYPE.keys())
for correct_type, pokemons in POKEMONS_BY_TYPE.items():
    wrong_types = all_types.copy().pop(all_types.index(correct_type))
    new_testcases = [
        WordGroup(base_word=pokemon, close_word=correct_type, far_word=wrong_type)
        for pokemon, wrong_type in product(pokemons, wrong_types)
    ]
    test_cases.extend(new_testcases)

#@pytest.mark.parametrize("test_case", test_cases)
@pytest.mark.parametrize("test_case", [WordGroup("charmander", "fire", "grass")])
def test_pokembedding(model: Word2Vec, test_case: WordGroup):
    """Asser that the close word is indeed closer."""
    close_similarity = model.compute_similarity(test_case.base_word, test_case.close_word)
    far_similarity = model.compute_similarity(test_case.base_word, test_case.far_word)
    assert (
        close_similarity < far_similarity
    ), f"{test_case.base_word}, {test_case.close_word}, {test_case.far_word} failed..."
