"""In this test we verify that the pokemons are closer to their type than to any other type."""

import json
from itertools import product
from typing import List

from .model import Word2Vec

with open("../data/eval_data_poke_and_types.json", "r") as f:
    POKEMONS_BY_TYPE = json.load(f)

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

    @classmethod 
    def tot_num_tests(cls) -> int:
        """Return the total number of tests."""
        return cls.pass_counter + cls.fail_counter

    @classmethod
    def pass_rate(cls) -> float:
        """Return the pass rate."""
        return float(cls.pass_counter) / cls.tot_num_tests()
    
    @classmethod
    def fail_rate(cls) -> float:
        """Return the fail rate."""
        return float(cls.fail_counter) / cls.tot_num_tests()


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


def entire_triplet_in_vocab(model: Word2Vec, triplet: WordTriplet) -> bool:
    """Check if all words in the triplet are in the vocabulary."""
    return (
        triplet.base_word in model.vocabulary
        and triplet.similar_word in model.vocabulary
        and triplet.dissimilar_word in model.vocabulary
    )


def test_script(model: Word2Vec):
    """Assemble all test triplets, filter out those not in the vocabulary, and compute the similarity."""
    triplets = assemble_test_triples()

    for triplet in triplets:
        if not entire_triplet_in_vocab(model, triplet):
            continue
        
        TestResults.register_result(
            triplet.base_word,
            triplet.similar_word,
            triplet.dissimilar_word,
            model.compute_similarity(triplet.base_word, triplet.similar_word),
            model.compute_similarity(triplet.base_word, triplet.dissimilar_word),
        )

    print(
        f"Pass: {TestResults.pass_rate():.1%} ({TestResults.pass_counter} of {TestResults.tot_num_tests()})\n",
        f"Fail: {TestResults.fail_rate():.1%} ({TestResults.fail_counter} of {TestResults.tot_num_tests()})",
    )
    return TestResults
