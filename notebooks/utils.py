import json
import re
import string
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Union

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE

from model import Word2Vec

# Define default Pokemon type colors
DEFAULT_TYPE_COLORS = {
    "fire": "red",
    "water": "blue",
    "electric": "yellow",
    "grass": "green", 
    "psychic": "purple",
    "normal": "gray",
    "fighting": "brown",
}

def _project_embeddings(embeddings: np.ndarray, dim: int, **kwargs) -> np.ndarray: #rnd_seed: int = None) -> np.ndarray:
    """Project embeddings onto a lower-dimensional space."""
    tsne = TSNE(n_components=dim, **kwargs)
    return tsne.fit_transform(embeddings)


def visualize_embeddings(
    model: Word2Vec, 
    dim: int = 2, 
    words: Optional[Union[List[str], Dict[str, str]]] = None,
    title: str = "projected embeddings",
    color_map: Optional[Dict[str, str]] = DEFAULT_TYPE_COLORS,
    **kwargs,
):
    """Visualize the embeddings in a 2D or 3D space.
    
    Args:
        model (Word2Vec): The Word2Vec model to visualize.
        dim (int, optional): Dimension of the projection. Defaults to 2.
        words (Optional[Union[List[str], Dict[str, str]]], optional): 
            - If List[str]: List of words to visualize (all assigned same class).
            - If Dict[str, str]: Dictionary where keys are words and values are classes/categories.
              Each class will receive a distinct color in the plot.
            - If None: All words in the vocabulary are visualized.
        title (str, optional): Title for the plot. Defaults to "projected embeddings".
        color_map (Optional[Dict[str, str]], optional): Dictionary mapping class names to specific colors.
            Default colors are used for Pokemon types: fire=red, water=blue, electric=yellow, 
            grass=green, psychic=purple.
        **kwargs: Additional keyword arguments passed to the TSNE constructor (e.g. random_state: int)
        
    Returns:
        pd.DataFrame: DataFrame containing the projection data.
    """
    if dim not in (2, 3):
        raise ValueError("dim must be 2 or 3.")
    
    if words is not None:
        # Standardize input: convert list to dictionary if needed
        if isinstance(words, list):
            words = {word: "default" for word in words}
        # Validate words are in vocabulary
        if set(words.keys()) - model.vocabulary:
            raise ValueError("Some words are not in the vocabulary.")
    
    # Get embeddings and project them
    if words:
        # Process words from dictionary
        word_list = list(words.keys())
        class_list = list(words.values())
        embeddings = np.array([model.get_word_embedding(word) for word in word_list])
        projected_embeddings = _project_embeddings(embeddings, dim, **kwargs)
        labels = word_list
        classes = class_list
    else:
        # Retrieve all embeddings in the vocabulary
        embeddings = model.get_layer("word_embedding").get_weights()[0]
        projected_embeddings = _project_embeddings(embeddings, dim, **kwargs)
        labels = [model.idx2word[idx] for idx in range(model.vocabulary_size)]
        # Assign all vocabulary to the same class for consistent coloring
        classes = ["vocabulary"] * len(labels)

    # Create DataFrame with projection data
    df = pd.DataFrame(projected_embeddings, columns=["x", "y"] if dim == 2 else ["x", "y", "z"])
    df["label"] = labels
    df["class"] = classes
    
    # Plot based on dimension
    if dim == 2:
        fig = px.scatter(
            df, 
            x="x", 
            y="y", 
            hover_name="label", 
            color="class", 
            title=title,
            color_discrete_map=color_map  # Use our color map
        )
        fig.update_traces(marker=dict(size=8, opacity=0.8))
        fig.show()
        
    elif dim == 3:
        fig = px.scatter_3d(
            df,
            x="x",
            y="y",
            z="z",
            hover_name="label",
            color="class",
            title=title,
            color_discrete_map=color_map  # Use our color map
        )
        fig.update_traces(marker=dict(size=5, opacity=0.8))
        fig.show()


# Clean and tokenize text
def _pre_process_sentence(sentece: str) -> List[str]:
    """Lowercase, remove punctuation, and split into words."""
    sentece = sentece.lower()  
    sentece = re.sub(f"[{string.punctuation}]", "", sentece)  
    words = sentece.split()  
    return words


def import_dataset(
    dataset_path: Path,
    pre_process: bool = True,
) -> List[List[str]]:
    """Import a dataset from a JSON file."""
    with open(dataset_path) as f:
        json_data = json.load(f)
    if pre_process:
        return [_pre_process_sentence(sentence) for sentence in json_data["data"]]
    else:
        return json_data["data"]


def merge_datasets(
    *datasets: List[List[str]],
) -> List[List[str]]:
    """Concatenate all datasets together."""
    return [sentence for dataset in datasets for sentence in dataset]
    
