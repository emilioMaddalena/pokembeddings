from collections import defaultdict
from typing import List

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dot, Embedding
from tensorflow.keras.models import Model


class Word2Vec(Model):  # noqa: D101
    """A Word2Vec model implementation using Keras.

    To use it, follow the steps:
    - Instantiate it with a list of tokenized sentences.
    - Call self.prepare_dataset to create the trainig batches.
    - Call self.compile() to specify the optimizer and loss.
    - Train the model using self.fit().
    - Call self.get_word_embedding() and self.compute_similarity() to have fun.
    """

    def __init__(
            self, 
            dataset: List[List[str]], 
            embedding_dim: int, 
            weight_tying: bool = False
        ):
        """Initialize the base attributes.

        Save the dataset, extract vocabulary and index the tokens.
        Define the base layers for the model.
        Define the similarity metric, hardcoded cosine.

        Args:
            dataset: A set of tokenized sentences on which the model will be trained.
            embedding_dim: The dimension of the embedding space.
            weight_tying: If the context and center embeddings should share weights.
        """
        super(Word2Vec, self).__init__()

        # Extract vocabulary from the dataset
        self.dataset = dataset
        self.vocabulary = set(word for sentence in dataset for word in sentence)

        # Create auxiliary mappings
        self.vocabulary_indexes = list(range(self.vocabulary_size))
        self.word2idx = {word: idx for idx, word in enumerate(self.vocabulary)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

        # Setup the main components
        self.embedding_dim = embedding_dim
        self.center_embedding = Embedding(self.vocabulary_size, self.embedding_dim, name="word_embedding")
        if weight_tying:
            self.context_embedding = self.center_embedding
        else:
            self.context_embedding = Embedding(self.vocabulary_size, self.embedding_dim)
        # Cosine similarity
        self.similarity_metric = Dot(axes=1, normalize=True)

    @property
    def vocabulary_size(self) -> int:
        """Return the size of the vocabulary."""
        return len(self.vocabulary)

    def prepare_dataset(self, window_size: int, num_negative_samples: int = 5) -> tf.Tensor:
        """Transform the dataset so that it can be used for training.

        Generate (center, context) word pairs. The dataset is then shuffled and batched.
        Both positive and negative samples are used.
        """
        positive_pairs = []
        for sentence in self.dataset:
            for idx, center_word in enumerate(sentence):
                context_start = max(0, idx - window_size)
                context_end = min(len(sentence), idx + window_size + 1)
                for context_idx in range(context_start, context_end):
                    if context_idx != idx:  # Avoid self-pairing
                        positive_pairs.append(
                            (self.word2idx[center_word], self.word2idx[sentence[context_idx]])
                        )

        print(
            "Sample word pairs:",
            [(self.idx2word[c], self.idx2word[ctx]) for c, ctx in positive_pairs[:5]],
        )

        # Create dictionary with lists instead of sets initially
        center_to_context = defaultdict(set)
        for center_idx, context_idx in positive_pairs:
            center_to_context[center_idx].add(context_idx)

        all_centers = []
        all_contexts = []
        all_labels = []

        # Generate positive samples (label=1.0)
        for center_idx, context_idx in positive_pairs:
            all_centers.append(center_idx)
            all_contexts.append(context_idx)
            all_labels.append(1.0)

        # Generate negative samples (label=0.0)
        for center_idx in center_to_context:
            valid_negative_idxs = list(
                set(self.vocabulary_indexes) - set(center_to_context[center_idx])
            )
            if not valid_negative_idxs:
                continue
            for _ in range(num_negative_samples):
                negative_idx = np.random.choice(valid_negative_idxs)
                all_centers.append(center_idx)
                all_contexts.append(negative_idx)
                all_labels.append(0.0)

        # Convert to arrays
        all_centers = np.array(all_centers, dtype=np.int32)
        all_contexts = np.array(all_contexts, dtype=np.int32)
        labels = np.array(all_labels, dtype=np.float32)

        # Create a TensorFlow dataset
        train_dataset = tf.data.Dataset.from_tensor_slices(
            ({"center": all_centers, "context": all_contexts}, labels)
        )
        train_dataset = train_dataset.shuffle(10000).batch(128)

        return train_dataset

    def call(self, inputs):  
        """Forward pass of the model.

        Args:
            inputs: A dictionary containing the center and context word indices.

        Returns:
            probability: The probability of the context word given the center word.
        """
        center_indices = inputs["center"]
        context_indices = inputs["context"]
        # Get embeddings
        center_embedding = self.center_embedding(center_indices)
        context_embedding = self.context_embedding(context_indices)
        # Calculate similarity
        similarity_score = self.similarity_metric([center_embedding, context_embedding])
        # Turn -1, 1 into 0, 1
        probability = (1 + similarity_score) / 2
        return probability

    def get_word_embedding(self, word: str) -> np.ndarray:
        """Get the embedding of a single word."""
        if word not in self.vocabulary:
            raise ValueError(f"{word} not in vocabulary!")
        # Get index in the vocabulary
        word_index = self.word2idx[word]
        # Get embeding matrix
        embedding_matrix = self.get_layer("word_embedding").get_weights()[0]
        return embedding_matrix[word_index]

    def compute_similarity(self, word_a: str, word_b: str):
        """Compute similarity between two single words.
        
        This method uses the center embedding matrix only. It's supposed to be used
        for testing purposes only.

        This is NOT equivalent to using calling the model as that would yield probabilities
        instead of similarity scores.
        """
        embedding_a = tf.reshape(self.get_word_embedding(word_a), (1, self.embedding_dim))
        embedding_b = tf.reshape(self.get_word_embedding(word_b), (1, self.embedding_dim))
        return self.similarity_metric([embedding_a, embedding_b]).numpy()[0, 0]

    # The following two methods are needed to ensure Keras will
    # correctly save and load not only the base model, but also
    # all custom attributes
    def get_config(self):  # noqa: D102
        config = super(Word2Vec, self).get_config()
        config.update(
            {
                "dataset": self.dataset,
                "embedding_dim": self.embedding_dim,
                "vocabulary": list(self.vocabulary),  # Convert set to list for serialization
                "word2idx": self.word2idx,
                "idx2word": {
                    str(k): v for k, v in self.idx2word.items()
                },  # Convert int keys to str
            }
        )
        return config

    @classmethod
    def from_config(cls, config):  # noqa: D102
        # Rebuild the model with the same parameters
        dataset = config.pop("dataset")
        embedding_dim = config.pop("embedding_dim")
        instance = cls(dataset, embedding_dim)
        # Restore custom attributes if they exist in the config
        if "vocabulary" in config:
            instance.vocabulary = set(config.pop("vocabulary"))  # Convert back to set
        if "word2idx" in config:
            instance.word2idx = config.pop("word2idx")
        if "idx2word" in config:
            instance.idx2word = {int(k): v for k, v in config.pop("idx2word").items()}
        return instance
