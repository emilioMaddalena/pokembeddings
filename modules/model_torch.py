from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class Word2VecDataset(Dataset):
    """PyTorch dataset for Word2Vec training."""
    
    def __init__(self, centers, contexts, labels):
        """Initialize with arrays of center words, context words, and labels."""
        self.centers = torch.tensor(centers, dtype=torch.long)
        self.contexts = torch.tensor(contexts, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float)
        
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.centers)
    
    def __getitem__(self, idx):
        """Return a single sample as a dictionary of inputs and its label."""
        return {
            'center': self.centers[idx], 
            'context': self.contexts[idx]
        }, self.labels[idx]


class Word2Vec(nn.Module):
    """A Word2Vec model implementation using PyTorch.

    To use it, follow the steps:
    - Instantiate it with a list of tokenized sentences.
    - Call self.prepare_dataset to create the training batches.
    - Train the model using self.fit().
    - Call self.get_word_embedding() and self.compute_similarity() to have fun.
    """

    def __init__(self, dataset: List[List[str]], embedding_dim: int, weight_tying: bool = False):
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
        self.vocabulary_size = len(self.vocabulary)
        self.vocabulary_indexes = list(range(self.vocabulary_size))
        self.word2idx = {word: idx for idx, word in enumerate(self.vocabulary)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

        # Setup the main components
        self.embedding_dim = embedding_dim
        self.center_embedding = nn.Embedding(self.vocabulary_size, self.embedding_dim)
        
        if weight_tying:
            self.context_embedding = self.center_embedding
        else:
            self.context_embedding = nn.Embedding(self.vocabulary_size, self.embedding_dim)
        
        # We'll implement cosine similarity in the forward pass

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
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
        
        # Calculate cosine similarity
        center_norm = torch.norm(center_embedding, dim=1, keepdim=True)
        context_norm = torch.norm(context_embedding, dim=1, keepdim=True)
        dot_product = torch.sum(center_embedding * context_embedding, dim=1)
        
        # Avoid division by zero
        norm_product = torch.clamp(center_norm * context_norm, min=1e-8).squeeze()
        similarity_score = dot_product / norm_product
        
        # Turn [-1, 1] into [0, 1]
        probability = (1 + similarity_score) / 2
        
        return probability

    def prepare_dataset(self, window_size: int, num_negative_samples: int = 5, batch_size: int = 128) -> DataLoader:
        """Transform the dataset so that it can be used for training.

        Generate (center, context) word pairs. The dataset is then shuffled and batched.
        Both positive and negative samples are used.
        
        Returns:
            DataLoader: PyTorch DataLoader for training
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

        # Create PyTorch dataset and dataloader
        dataset = Word2VecDataset(
            centers=all_centers,
            contexts=all_contexts,
            labels=all_labels
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        return dataloader

    def fit(self, dataloader: DataLoader, epochs: int = 10, lr: float = 0.001) -> Dict[str, List[float]]:
        """Train the model.
        
        Args:
            dataloader: PyTorch DataLoader containing training examples
            epochs: Number of training epochs
            lr: Learning rate
            
        Returns:
            Dict containing training history
        """
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.BCELoss()
        
        history = {
            'loss': [],
            'accuracy': []
        }
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            correct_predictions = 0
            total_predictions = 0
            
            for batch_idx, (inputs, labels) in enumerate(dataloader):
                # Forward pass
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Track metrics
                epoch_loss += loss.item()
                predicted = (outputs > 0.5).float()
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
            
            # Calculate epoch metrics
            avg_loss = epoch_loss / len(dataloader)
            accuracy = correct_predictions / total_predictions
            
            # Store metrics
            history['loss'].append(avg_loss)
            history['accuracy'].append(accuracy)
            
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.4f}")
        
        return history

    def get_word_embedding(self, word: str) -> np.ndarray:
        """Get the embedding of a single word."""
        if word not in self.vocabulary:
            raise ValueError(f"{word} not in vocabulary!")
        
        # Get index in the vocabulary
        word_index = self.word2idx[word]
        
        # Get embedding vector
        with torch.no_grad():
            embedding = self.center_embedding(torch.tensor(word_index)).numpy()
        
        return embedding

    def compute_similarity(self, word_a: str, word_b: str) -> float:
        """Compute similarity between two single words."""
        if word_a not in self.vocabulary or word_b not in self.vocabulary:
            raise ValueError(f"Words must be in vocabulary")
        
        # Get indices
        idx_a = self.word2idx[word_a]
        idx_b = self.word2idx[word_b]
        
        # Get embeddings
        with torch.no_grad():
            emb_a = self.center_embedding(torch.tensor(idx_a)).view(1, -1)
            emb_b = self.center_embedding(torch.tensor(idx_b)).view(1, -1)
            
            # Compute cosine similarity
            a_norm = torch.norm(emb_a)
            b_norm = torch.norm(emb_b)
            dot_product = torch.sum(emb_a * emb_b)
            similarity = dot_product / (a_norm * b_norm)
            
        return similarity.item()

    def save(self, path: str):
        """Save model to a file."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'embedding_dim': self.embedding_dim,
            'vocabulary': list(self.vocabulary),
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'dataset': self.dataset
        }, path)
    
    @classmethod
    def load(cls, path: str, map_location=None):
        """Load model from a file."""
        checkpoint = torch.load(path, map_location=map_location)
        
        # Create a new model instance
        model = cls(
            dataset=checkpoint['dataset'],
            embedding_dim=checkpoint['embedding_dim']
        )
        
        # Restore state
        model.vocabulary = set(checkpoint['vocabulary'])
        model.word2idx = checkpoint['word2idx']
        model.idx2word = checkpoint['idx2word']
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model
