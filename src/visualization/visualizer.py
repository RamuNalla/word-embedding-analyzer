"""
Visualization utilities for word embeddings.
"""

from typing import List, Optional, Dict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
import logging

from ..embeddings.base_embedding import BaseEmbedding

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingVisualizer:
    """Visualizer for word embeddings."""

    def __init__(self, embedding: BaseEmbedding):
        """
        Initialize visualizer.
        
        Args:
            embedding: Embedding model to visualize
        """
        self.embedding = embedding


    def reduce_dimensions(self, words: List[str], method: str = 'tsne', 
                         n_components: int = 2, **kwargs) -> np.ndarray:
        """
        Reduce embedding dimensions for visualization.
        
        Args:
            words: List of words to visualize
            method: Dimensionality reduction method ('tsne' or 'pca')
            n_components: Number of dimensions (2 or 3)
            **kwargs: Additional arguments for the reduction method
            
        Returns:
            Reduced dimension vectors
        """
        # Get vectors for all words
        vectors = []
        valid_words = []
        
        for word in words:
            vec = self.embedding.get_vector(word)
            if vec is not None:
                vectors.append(vec)
                valid_words.append(word)
        
        if not vectors:
            raise ValueError("No valid word vectors found")
        
        vectors = np.array(vectors)
        
        # Apply dimensionality reduction
        if method.lower() == 'tsne':
            perplexity = kwargs.get('perplexity', min(30, len(vectors) - 1))
            reducer = TSNE(n_components=n_components, perplexity=perplexity, 
                          random_state=kwargs.get('random_state', 42),
                          n_iter=kwargs.get('n_iter', 1000))
        elif method.lower() == 'pca':
            reducer = PCA(n_components=n_components, 
                         random_state=kwargs.get('random_state', 42))
        else:
            raise ValueError(f"Unknown method: {method}")
        
        reduced = reducer.fit_transform(vectors)
        
        return reduced, valid_words
    
    def plot_words_2d(self, words: List[str], method: str = 'tsne', 
                     figsize: tuple = (12, 8), title: Optional[str] = None,
                     save_path: Optional[str] = None, **kwargs):
        """
        Plot words in 2D space.
        
        Args:
            words: List of words to plot
            method: Dimensionality reduction method
            figsize: Figure size
            title: Plot title
            save_path: Path to save figure
            **kwargs: Additional arguments for reduction method
        """
        reduced, valid_words = self.reduce_dimensions(words, method, n_components=2, **kwargs)
        
        plt.figure(figsize=figsize)
        plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.6, s=100)
        
        # Add labels
        for i, word in enumerate(valid_words):
            plt.annotate(word, (reduced[i, 0], reduced[i, 1]), 
                        fontsize=10, alpha=0.8)
        
        if title is None:
            title = f"Word Embeddings - {method.upper()} Visualization"
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel(f'{method.upper()} Component 1')
        plt.ylabel(f'{method.upper()} Component 2')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_words_3d_interactive(self, words: List[str], method: str = 'tsne',
                                 title: Optional[str] = None, **kwargs):
        """
        Create interactive 3D plot of words using Plotly.
        
        Args:
            words: List of words to plot
            method: Dimensionality reduction method
            title: Plot title
            **kwargs: Additional arguments for reduction method
            
        Returns:
            Plotly figure object
        """
        reduced, valid_words = self.reduce_dimensions(words, method, n_components=3, **kwargs)
        
        fig = go.Figure(data=[go.Scatter3d(
            x=reduced[:, 0],
            y=reduced[:, 1],
            z=reduced[:, 2],
            mode='markers+text',
            text=valid_words,
            textposition='top center',
            marker=dict(
                size=8,
                color=np.arange(len(valid_words)),
                colorscale='Viridis',
                showscale=True
            )
        )])
        
        if title is None:
            title = f"Word Embeddings - 3D {method.upper()} Visualization"
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=f'{method.upper()} 1',
                yaxis_title=f'{method.upper()} 2',
                zaxis_title=f'{method.upper()} 3'
            ),
            height=700
        )
        
        return fig
    

    def plot_similarity_heatmap(self, words: List[str], figsize: tuple = (10, 8),
                               title: Optional[str] = None,
                               save_path: Optional[str] = None):
        """
        Plot similarity heatmap for a list of words.
        
        Args:
            words: List of words
            figsize: Figure size
            title: Plot title
            save_path: Path to save figure
        """
        from ..tasks.similarity import similarity_matrix
        
        # Calculate similarity matrix
        sim_matrix = similarity_matrix(self.embedding, words)
        
        plt.figure(figsize=figsize)
        sns.heatmap(sim_matrix, xticklabels=words, yticklabels=words,
                   annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                   square=True, linewidths=0.5)
        
        if title is None:
            title = "Word Similarity Heatmap"
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()

    def plot_analogy_visualization(self, positive: List[str], negative: List[str],
                                  result_words: List[str], method: str = 'tsne',
                                  figsize: tuple = (12, 8), 
                                  save_path: Optional[str] = None):
        """
        Visualize word analogy in 2D space.
        
        Args:
            positive: Positive words in analogy
            negative: Negative words in analogy
            result_words: Predicted result words
            method: Dimensionality reduction method
            figsize: Figure size
            save_path: Path to save figure
        """
        all_words = positive + negative + result_words
        reduced, valid_words = self.reduce_dimensions(all_words, method, n_components=2)
        
        # Create word to coordinate mapping
        word_coords = {word: reduced[i] for i, word in enumerate(valid_words)}
        
        plt.figure(figsize=figsize)
        
        # Plot positive words
        for word in positive:
            if word in word_coords:
                coords = word_coords[word]
                plt.scatter(coords[0], coords[1], c='green', s=200, alpha=0.6, 
                          marker='o', edgecolors='black', linewidth=2, label='Positive')
                plt.annotate(word, coords, fontsize=12, fontweight='bold')
        
        # Plot negative words
        for word in negative:
            if word in word_coords:
                coords = word_coords[word]
                plt.scatter(coords[0], coords[1], c='red', s=200, alpha=0.6,
                          marker='s', edgecolors='black', linewidth=2, label='Negative')
                plt.annotate(word, coords, fontsize=12, fontweight='bold')
        
        # Plot result words
        for word in result_words:
            if word in word_coords:
                coords = word_coords[word]
                plt.scatter(coords[0], coords[1], c='blue', s=200, alpha=0.6,
                          marker='^', edgecolors='black', linewidth=2, label='Result')
                plt.annotate(word, coords, fontsize=12, fontweight='bold')
        
        # Remove duplicate labels
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), fontsize=10)
        
        plt.title(f"Analogy Visualization: {' + '.join(positive)} - {' - '.join(negative)}", 
                 fontsize=14, fontweight='bold')
        plt.xlabel(f'{method.upper()} Component 1')
        plt.ylabel(f'{method.upper()} Component 2')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    

    def plot_word_neighbors(self, word: str, topn: int = 10, method: str = 'tsne',
                           figsize: tuple = (12, 8), save_path: Optional[str] = None):
        """
        Visualize a word and its nearest neighbors.
        
        Args:
            word: Target word
            topn: Number of neighbors to show
            method: Dimensionality reduction method
            figsize: Figure size
            save_path: Path to save figure
        """
        # Get similar words
        similar_words = self.embedding.most_similar(word, topn=topn)
        words = [word] + [w for w, _ in similar_words]
        
        reduced, valid_words = self.reduce_dimensions(words, method, n_components=2)
        
        plt.figure(figsize=figsize)
        
        # Plot target word
        plt.scatter(reduced[0, 0], reduced[0, 1], c='red', s=300, alpha=0.8,
                   marker='*', edgecolors='black', linewidth=2, label='Target Word')
        plt.annotate(valid_words[0], (reduced[0, 0], reduced[0, 1]), 
                    fontsize=14, fontweight='bold', color='red')
        
        # Plot neighbors
        for i in range(1, len(reduced)):
            plt.scatter(reduced[i, 0], reduced[i, 1], c='blue', s=150, alpha=0.6)
            plt.annotate(valid_words[i], (reduced[i, 0], reduced[i, 1]), fontsize=10)
        
        plt.title(f"Word Neighbors: '{word}'", fontsize=14, fontweight='bold')
        plt.xlabel(f'{method.upper()} Component 1')
        plt.ylabel(f'{method.upper()} Component 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()