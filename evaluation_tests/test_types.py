"""In this test we verify that the pokemons are closer to their type than to any other type."""

from itertools import product
from typing import List

from model import Word2Vec

from .poke_types import POKEMONS_BY_TYPE


class TestResults:
    """This class stores and post-processes the test results."""  # noqa: D404

    pass_counter = 0
    fail_counter = 0

    @classmethod
    def register_result(
        cls,
        base_word: str,
        similar_word: str,
        dissimilar_word: str,
        base_to_similar: float,
        base_to_dissimilar: float,
    ):
        """Update the pass/fail counters."""
        if base_to_similar > base_to_dissimilar:
            TestResults.pass_counter += 1
        else:
            TestResults.fail_counter += 1


class WordTriplet:  # noqa: D101
    def __init__(self, base_word: str, similar_word: str, dissimilar_word: str):  # noqa: D107
        self.base_word = base_word
        self.similar_word = similar_word
        self.dissimilar_word = dissimilar_word


def assemble_test_triples() -> List:
    """TBW."""
    triplets = []
    all_types = list(POKEMONS_BY_TYPE.keys())

    # Create triples by pairing each pokemon with its type and all wrong type combinations
    for correct_type, pokemons in POKEMONS_BY_TYPE.items():
        wrong_types = set(all_types.copy()) - set([correct_type])
        triplets.extend(
            [
                WordTriplet(
                    base_word=pokemon, similar_word=correct_type, dissimilar_word=wrong_type
                )
                for pokemon, wrong_type in product(pokemons, wrong_types)
            ]
        )
    return triplets


def test_types(model: Word2Vec):
    """Asser that the close word is indeed closer."""
    triplets = assemble_test_triples()

    for triplet in triplets:
        # Check if the base word is in the vocabulary
        if triplet.base_word not in model.vocabulary:
            print(f"Word '{triplet.base_word}' not in vocabulary.")
            continue
        elif triplet.similar_word not in model.vocabulary:
            print(f"Word '{triplet.similar_word}' not in vocabulary.")
            continue
        elif triplet.dissimilar_word not in model.vocabulary:
            print(f"Word '{triplet.dissimilar_word}' not in vocabulary.")
            continue

        TestResults.register_result(
            triplet.base_word,
            triplet.similar_word,
            triplet.dissimilar_word,
            model.compute_similarity(triplet.base_word, triplet.similar_word),
            model.compute_similarity(triplet.base_word, triplet.dissimilar_word),
        )

    print(
        f"Pass: {TestResults.pass_counter/len(triplets):.1%} ({TestResults.pass_counter}) \n", 
        f"Fail: {TestResults.fail_counter/len(triplets):.1%} ({TestResults.fail_counter})"
    )
