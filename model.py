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

    def __init__(self, dataset: List[List[str]], embedding_dim: int):
        """_summary_.

        Args:
            dataset: A set of tokenized sentences on which the model will be trained.
            embedding_dim: The dimension of the embedding space.
        """
        super(Word2Vec, self).__init__()

        # Extract vocabulary from the dataset
        self.dataset = dataset
        vocabulary = set(word for sentence in dataset for word in sentence)
        self.vocabulary = vocabulary
        self.vocabulary_size = len(vocabulary)

        # Create auxiliary mappings
        word2idx = {word: idx for idx, word in enumerate(vocabulary)}
        idx2word = {idx: word for word, idx in word2idx.items()}
        self.word2idx = word2idx
        self.idx2word = idx2word

        print("Word2Vec vocabulary size:", len(vocabulary))
        print("Word2Vec vocabulary words and indexes:", word2idx)

        # Setup the main components
        self.embedding_dim = embedding_dim
        self.embedding = Embedding(self.vocabulary_size, self.embedding_dim, name="word_embedding")
        self.similarity_metric = Dot(axes=1, normalize=False)

    def prepare_dataset(self, window_size: int) -> tf.Tensor:
        """Generate (center, context) word pairs. The dataset is then shuffled and batched."""
        pairs = []
        for sentence in self.dataset:
            for idx, center_word in enumerate(sentence):
                context_start = max(0, idx - window_size)
                context_end = min(len(sentence), idx + window_size + 1)
                for context_idx in range(context_start, context_end):
                    if context_idx != idx:  # Avoid self-pairing
                        pairs.append(
                            (self.word2idx[center_word], self.word2idx[sentence[context_idx]])
                        )

        print("Sample word pairs:", [(self.idx2word[c], self.idx2word[ctx]) for c, ctx in pairs[:5]])

        # Extract center and context words as separate lists
        center_words, context_words = zip(*pairs)
        center_words = np.array(center_words, dtype=np.int32)
        context_words = np.array(context_words, dtype=np.int32)
        labels = np.ones(len(center_words), dtype=np.float32)  # Positive examples only

        # Create a TensorFlow dataset
        train_dataset = tf.data.Dataset.from_tensor_slices(((center_words, context_words), labels))
        train_dataset = train_dataset.map(lambda pair, label: ((tf.stack(pair), label), label)) 
        train_dataset = train_dataset.shuffle(10000).batch(128)

        for element in train_dataset.take(1):
            batch, label = element
            print("Batch shape:", batch[0].shape)
            print("Batch contents:", batch[0].numpy())
            print("Labels:", label.numpy())

        return train_dataset

    def call(self, inputs):  # noqa: D102
        pair, label = inputs
        pair = tf.reshape(pair, (-1, 2))
        center_embedding = self.embedding(pair[:, 0])
        context_embedding = self.embedding(pair[:, 1])
        similarity_score = self.similarity_metric([center_embedding, context_embedding])
        return similarity_score

    def get_word_embedding(self, word: str) -> np.nparray:  
        """Get the embedding of a single word."""
        if word not in self.vocabulary:
            raise ValueError(f"{word} not in vocabulary!")
        # Get index in the vocabulary
        word_index = self.word2idx[word]
        # Get embeding matrix
        embedding_matrix = self.get_layer("word_embedding").get_weights()[0]
        return embedding_matrix[word_index]

    def compute_similarity(self, word_a: str, word_b: str):
        """Compute similarity between two single words."""
        embedding_a = tf.reshape(self.get_word_embedding(word_a), (1, self.embedding_dim))
        embedding_b = tf.reshape(self.get_word_embedding(word_b), (1, self.embedding_dim))
        return self.similarity_metric([embedding_a, embedding_b]).numpy()[0, 0]
