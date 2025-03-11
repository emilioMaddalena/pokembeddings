from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE

from model import Word2Vec


def _project_embeddings(embeddings: np.ndarray, dim: int, rnd_seed: int) -> np.ndarray:
    """Project embeddings onto a lower-dimensional space."""
    tsne = TSNE(n_components=dim, random_state=rnd_seed)
    return tsne.fit_transform(embeddings)


def visualize_embeddings(
    model: Word2Vec, dim: int = 2, rnd_seed: int = 123, words: Optional[List[str]] = None
):
    """Visualize the embeddings in a 2D or 3D space.
    
    Args:
        model (Word2Vec): The Word2Vec model to visualize.
        dim (int, optional): Dimension of the projection. Defaults to 2.
        rnd_seed (int, optional): Random seed for the projection. Defaults to 123.
        words (Optional[List[str]], optional): List of words to visualize. If None,
            all words in the vocabulary are visualized. Defaults to None.
    """
    if dim not in (2, 3):
        raise ValueError("dim must be 2 or 3.")
    if words:
        if set(words) - model.vocabulary:
            raise ValueError("Some words are not in the vocabulary.")

    # Get embeddings and project them
    if words:
        # process all provided words
        embeddings = np.array([model.get_word_embedding(word) for word in words])
        projected_embeddings = _project_embeddings(embeddings, dim, rnd_seed)
        labels = words
    else:
        # then retrieve all embeddings in the vocabulary
        embeddings = model.get_layer("word_embedding").get_weights()[0]
        projected_embeddings = _project_embeddings(embeddings, dim, rnd_seed)
        labels = [model.idx2word[idx] for idx in range(model.vocabulary_size)]

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
