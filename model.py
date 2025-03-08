from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import tensorflow as tf
from sklearn.manifold import TSNE
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
        """Initialize the base attributes.

        Save the dataset, extract vocabulary and index the tokens.
        Define the base layers for the model.

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
        self.similarity_metric = Dot(axes=1, normalize=True) # cosine similarity

    def prepare_dataset(self, window_size: int) -> tf.Tensor:
        """Transform the dataset so that it can be used for training.

        Generate (center, context) word pairs. The dataset is then shuffled and batched.
        """
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

        print(
            "Sample word pairs:", [(self.idx2word[c], self.idx2word[ctx]) for c, ctx in pairs[:5]]
        )

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
        """Compute similarity between two single words."""
        embedding_a = tf.reshape(self.get_word_embedding(word_a), (1, self.embedding_dim))
        embedding_b = tf.reshape(self.get_word_embedding(word_b), (1, self.embedding_dim))
        return self.similarity_metric([embedding_a, embedding_b]).numpy()[0, 0]

    def _project_embeddings(self, embeddings: np.ndarray, dim: int, rnd_seed: int) -> np.ndarray:
        """Project embeddings onto a lower-dimensional space."""
        tsne = TSNE(n_components=dim, random_state=rnd_seed)
        return tsne.fit_transform(embeddings)

    def visualize_embeddings(
        self, dim: int = 2, rnd_seed: int = 123, words: Optional[List[str]] = None
    ):
        """Visualize the embeddings in a 2D or 3D space."""
        if dim not in (2, 3):
            raise ValueError("dim must be 2 or 3.")
        if words:
            if set(words) - self.vocabulary:
                raise ValueError("Some words are not in the vocabulary.")

        # Get embeddings and project them
        if words:
            # process all provided words
            embeddings = np.array([self.get_word_embedding(word) for word in words])
            projected_embeddings = self._project_embeddings(embeddings, dim, rnd_seed)
            labels = words
        else:
            # then retrieve all embeddings in the vocabulary
            embeddings = self.get_layer("word_embedding").get_weights()[0]
            projected_embeddings = self._project_embeddings(embeddings, dim, rnd_seed)
            labels = [self.idx2word[idx] for idx in range(self.vocabulary_size)]

        # Plot
        if dim == 2:
            df = pd.DataFrame(projected_embeddings, columns=["x", "y"])
            df["label"] = labels
            fig = px.scatter(df, x="x", y="y", hover_name="label", title="projected embeddings")
            fig.update_traces(marker=dict(size=8, opacity=0.8))
            fig.show()
        elif dim == 3:
            df = pd.DataFrame(projected_embeddings, columns=["x", "y", "z"])
            df["label"] = labels
            fig = px.scatter_3d(
                df,
                x="x",
                y="y",
                z="z",
                hover_name="label",
                title="projected embeddings",
            )
            fig.update_traces(marker=dict(size=5, opacity=0.8))
            fig.show()

    # The following two methods are needed to ensure Keras will
    # correctly save and load not only the base model, but also
    # all custom attributes
    def get_config(self):  # noqa: D102
        config = super(Word2Vec, self).get_config()
        config.update(
            {
                "dataset": self.dataset,
                "embedding_dim": self.embedding_dim,
                "vocabulary": list(self.vocabulary),
                "vocabulary_size": self.vocabulary_size,
                "word2idx": self.word2idx,
                "idx2word": self.idx2word,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):  # noqa: D102
        dataset = config.pop("dataset")
        embedding_dim = config.pop("embedding_dim")
        model = cls(dataset, embedding_dim)
        model.vocabulary = set(config.pop("vocabulary"))
        model.vocabulary_size = config.pop("vocabulary_size")
        model.word2idx = config.pop("word2idx")
        model.idx2word = config.pop("idx2word")
        return model
