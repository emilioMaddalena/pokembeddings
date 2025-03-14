"""
In this test we verify that the pokemons are closer to their type than to any other type.
""" 

from itertools import product
from typing import List

from model import Word2Vec

from .poke_types import POKEMONS_BY_TYPE


class WordGroup:  # noqa: D101
    def __init__(self, base_word: str, close_word: str, far_word: str):  # noqa: D107
        self.base_word = base_word
        self.close_word = close_word
        self.far_word = far_word


def all_test_cases() -> List:
    """Assemble all test cases from the raw test data.

    All pokemons are combined with their types and all other types.
    """
    test_cases = []
    all_types = list(POKEMONS_BY_TYPE.keys())

    for correct_type, pokemons in POKEMONS_BY_TYPE.items():
        wrong_types = all_types.copy().pop(all_types.index(correct_type))
        new_testcases = [
            WordGroup(base_word=pokemon, close_word=correct_type, far_word=wrong_type)
            for pokemon, wrong_type in product(pokemons, wrong_types)
        ]
        test_cases.extend(new_testcases)
    return test_cases


def test_types(model: Word2Vec, test_case: WordGroup):
    """Asser that the close word is indeed closer."""
    close_similarity = model.compute_similarity(test_case.base_word, test_case.close_word)
    far_similarity = model.compute_similarity(test_case.base_word, test_case.far_word)
    assert (
        close_similarity < far_similarity
    )
