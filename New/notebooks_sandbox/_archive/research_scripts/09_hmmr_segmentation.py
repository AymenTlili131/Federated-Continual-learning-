"""
09_hmmr_segmentation.py

Python implementation of Hidden Markov Model Regression (HMMR) for time-series
segmentation of predicted weights.

Based on: https://github.com/fchamroukhi/HMMR_r

This module applies HMMR to segment weight sequences and cluster subsequences
by their corresponding class labels.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent.parent))


class HMMR:
    """
    Hidden Markov Model Regression for time-series segmentation.
    
    Python implementation of the HMMR_r package functionality.
    """
    
    def __init__(
        self,
        n_states: int = 5,
        n_features: int = None,
        max_iter: int = 100,
        tol: float = 1e-4,
        random_state: int = 42
    ):
        """
        Initialize HMMR model.
        
        Args:
            n_states: Number of hidden states
            n_features: Number of features in observations
            max_iter: Maximum EM iterations
            tol: Convergence tolerance
            random_state: Random seed
        """
        self.n_states = n_states
        self.n_features = n_features
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        
        np.random.seed(random_state)
        
        # Model parameters
        self.transition_matrix = None  # (n_states, n_states)
        self.initial_probs = None      # (n_states,)
        self.regression_weights = None # (n_states, n_features, n_features)
        self.regression_bias = None    # (n_states, n_features)
        self.covariances = None        # (n_states, n_features, n_features)
        
        self.converged = False
        self.log_likelihood_history = []
    
    def _initialize_parameters(self, X: np.ndarray):
        """Initialize model parameters."""
        n_samples, n_features = X.shape
        
        if self.n_features is None:
            self.n_features = n_features
        
        # Initialize transition matrix (slightly favor staying in same state)
        self.transition_matrix = np.random.dirichlet(
            np.ones(self.n_states) * 0.5,
            size=self.n_states
        )
        np.fill_diagonal(self.transition_matrix, 0.7)
        self.transition_matrix = self.transition_matrix / self.transition_matrix.sum(axis=1, keepdims=True)
        
        # Initialize initial state probabilities
        self.initial_probs = np.ones(self.n_states) / self.n_states
        
        # Initialize regression parameters using K-means clustering
        kmeans = KMeans(n_clusters=self.n_states, random_state=self.random_state)
        labels = kmeans.fit_predict(X)
        
        self.regression_weights = np.zeros((self.n_states, n_features, n_features))
        self.regression_bias = np.zeros((self.n_states, n_features))
        self.covariances = np.zeros((self.n_states, n_features, n_features))
        
        for k in range(self.n_states):
            mask = labels == k
            if mask.sum() > 0:
                X_k = X[mask]
                self.regression_bias[k] = X_k.mean(axis=0)
                self.covariances[k] = np.cov(X_k.T) + np.eye(n_features) * 1e-6
            else:
                self.regression_bias[k] = X.mean(axis=0)
                self.covariances[k] = np.eye(n_features)
    
    def _forward_backward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Forward-backward algorithm for computing state probabilities.
        
        Returns:
            alpha: Forward probabilities (n_samples, n_states)
            beta: Backward probabilities (n_samples, n_states)
            log_likelihood: Log-likelihood of sequence
        """
        n_samples = len(X)
        
        # Compute emission probabilities
        emission_probs = np.zeros((n_samples, self.n_states))
        for k in range(self.n_states):
            try:
                emission_probs[:, k] = multivariate_normal.pdf(
                    X,
                    mean=self.regression_bias[k],
                    cov=self.covariances[k]
                )
            except:
                emission_probs[:, k] = 1e-10
        
        emission_probs = np.clip(emission_probs, 1e-10, None)
        
        # Forward pass
        log_alpha = np.zeros((n_samples, self.n_states))
        log_alpha[0] = np.log(self.initial_probs) + np.log(emission_probs[0])
        
        for t in range(1, n_samples):
            for j in range(self.n_states):
                log_alpha[t, j] = logsumexp(
                    log_alpha[t-1] + np.log(self.transition_matrix[:, j])
                ) + np.log(emission_probs[t, j])
        
        log_likelihood = logsumexp(log_alpha[-1])
        
        # Backward pass
        log_beta = np.zeros((n_samples, self.n_states))
        
        for t in range(n_samples - 2, -1, -1):
            for i in range(self.n_states):
                log_beta[t, i] = logsumexp(
                    np.log(self.transition_matrix[i]) +
                    np.log(emission_probs[t+1]) +
                    log_beta[t+1]
                )
        
        # Convert to probabilities
        alpha = np.exp(log_alpha - logsumexp(log_alpha, axis=1, keepdims=True))
        beta = np.exp(log_beta - logsumexp(log_beta, axis=1, keepdims=True))
        
        return alpha, beta, log_likelihood
    
    def _e_step(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        E-step: Compute state probabilities.
        
        Returns:
            gamma: State probabilities (n_samples, n_states)
            xi: Transition probabilities (n_samples-1, n_states, n_states)
        """
        alpha, beta, log_likelihood = self._forward_backward(X)
        
        # Compute gamma (state probabilities)
        gamma = alpha * beta
        gamma = gamma / gamma.sum(axis=1, keepdims=True)
        
        # Compute xi (transition probabilities)
        n_samples = len(X)
        xi = np.zeros((n_samples - 1, self.n_states, self.n_states))
        
        emission_probs = np.zeros((n_samples, self.n_states))
        for k in range(self.n_states):
            try:
                emission_probs[:, k] = multivariate_normal.pdf(
                    X,
                    mean=self.regression_bias[k],
                    cov=self.covariances[k]
                )
            except:
                emission_probs[:, k] = 1e-10
        
        emission_probs = np.clip(emission_probs, 1e-10, None)
        
        for t in range(n_samples - 1):
            for i in range(self.n_states):
                for j in range(self.n_states):
                    xi[t, i, j] = (
                        alpha[t, i] *
                        self.transition_matrix[i, j] *
                        emission_probs[t+1, j] *
                        beta[t+1, j]
                    )
            xi[t] = xi[t] / xi[t].sum()
        
        return gamma, xi
    
    def _m_step(self, X: np.ndarray, gamma: np.ndarray, xi: np.ndarray):
        """M-step: Update model parameters."""
        n_samples = len(X)
        
        # Update initial probabilities
        self.initial_probs = gamma[0]
        
        # Update transition matrix
        for i in range(self.n_states):
            for j in range(self.n_states):
                self.transition_matrix[i, j] = xi[:, i, j].sum() / gamma[:-1, i].sum()
        
        # Update regression parameters
        for k in range(self.n_states):
            gamma_k = gamma[:, k]
            gamma_sum = gamma_k.sum()
            
            if gamma_sum > 1e-6:
                # Update mean (regression bias)
                self.regression_bias[k] = (gamma_k[:, None] * X).sum(axis=0) / gamma_sum
                
                # Update covariance
                X_centered = X - self.regression_bias[k]
                self.covariances[k] = (
                    (gamma_k[:, None, None] * X_centered[:, :, None] * X_centered[:, None, :]).sum(axis=0) /
                    gamma_sum
                ) + np.eye(self.n_features) * 1e-6
    
    def fit(self, X: np.ndarray) -> 'HMMR':
        """
        Fit HMMR model using EM algorithm.
        
        Args:
            X: Time series data (n_samples, n_features)
        
        Returns:
            self
        """
        self._initialize_parameters(X)
        
        prev_log_likelihood = -np.inf
        
        for iteration in range(self.max_iter):
            # E-step
            gamma, xi = self._e_step(X)
            
            # M-step
            self._m_step(X, gamma, xi)
            
            # Compute log-likelihood
            _, _, log_likelihood = self._forward_backward(X)
            self.log_likelihood_history.append(log_likelihood)
            
            # Check convergence
            if abs(log_likelihood - prev_log_likelihood) < self.tol:
                self.converged = True
                break
            
            prev_log_likelihood = log_likelihood
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict most likely state sequence using Viterbi algorithm.
        
        Args:
            X: Time series data (n_samples, n_features)
        
        Returns:
            states: Most likely state sequence (n_samples,)
        """
        n_samples = len(X)
        
        # Compute emission probabilities
        emission_probs = np.zeros((n_samples, self.n_states))
        for k in range(self.n_states):
            try:
                emission_probs[:, k] = multivariate_normal.pdf(
                    X,
                    mean=self.regression_bias[k],
                    cov=self.covariances[k]
                )
            except:
                emission_probs[:, k] = 1e-10
        
        emission_probs = np.clip(emission_probs, 1e-10, None)
        
        # Viterbi algorithm
        log_delta = np.zeros((n_samples, self.n_states))
        psi = np.zeros((n_samples, self.n_states), dtype=int)
        
        # Initialization
        log_delta[0] = np.log(self.initial_probs) + np.log(emission_probs[0])
        
        # Recursion
        for t in range(1, n_samples):
            for j in range(self.n_states):
                temp = log_delta[t-1] + np.log(self.transition_matrix[:, j])
                psi[t, j] = np.argmax(temp)
                log_delta[t, j] = temp[psi[t, j]] + np.log(emission_probs[t, j])
        
        # Backtracking
        states = np.zeros(n_samples, dtype=int)
        states[-1] = np.argmax(log_delta[-1])
        
        for t in range(n_samples - 2, -1, -1):
            states[t] = psi[t+1, states[t+1]]
        
        return states
    
    def segment(self, X: np.ndarray) -> List[Tuple[int, int, int]]:
        """
        Segment time series into contiguous regions of same state.
        
        Args:
            X: Time series data (n_samples, n_features)
        
        Returns:
            segments: List of (start_idx, end_idx, state) tuples
        """
        states = self.predict(X)
        
        segments = []
        current_state = states[0]
        start_idx = 0
        
        for i in range(1, len(states)):
            if states[i] != current_state:
                segments.append((start_idx, i, current_state))
                current_state = states[i]
                start_idx = i
        
        # Add final segment
        segments.append((start_idx, len(states), current_state))
        
        return segments


def cluster_subsequences(
    segments: List[Tuple[int, int, int]],
    X: np.ndarray,
    labels: np.ndarray
) -> Dict[int, List[str]]:
    """
    Cluster weight subsequences to their corresponding class labels.
    
    Args:
        segments: List of (start, end, state) tuples
        X: Weight sequences
        labels: Class labels for each sequence
    
    Returns:
        state_to_classes: Mapping from state to list of class labels
    """
    state_to_classes = {}
    
    for start, end, state in segments:
        if state not in state_to_classes:
            state_to_classes[state] = []
        
        # Get labels for this segment
        segment_labels = labels[start:end]
        unique_labels = np.unique(segment_labels)
        
        state_to_classes[state].extend([str(label) for label in unique_labels])
    
    # Remove duplicates
    for state in state_to_classes:
        state_to_classes[state] = list(set(state_to_classes[state]))
    
    return state_to_classes


def visualize_segmentation(
    X: np.ndarray,
    segments: List[Tuple[int, int, int]],
    labels: np.ndarray,
    output_path: Path
):
    """Visualize time-series segmentation."""
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))
    
    # Plot first 3 dimensions of weights
    time_steps = np.arange(len(X))
    for i in range(min(3, X.shape[1])):
        axes[0].plot(time_steps, X[:, i], label=f'Dim {i+1}', alpha=0.7)
    
    # Add segment boundaries
    for start, end, state in segments:
        axes[0].axvline(x=start, color='red', linestyle='--', alpha=0.5)
        axes[0].text(
            (start + end) / 2, axes[0].get_ylim()[1] * 0.9,
            f'S{state}', ha='center', fontsize=10
        )
    
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Weight Value')
    axes[0].set_title('Weight Evolution with Segmentation')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot state sequence
    states = np.zeros(len(X))
    for start, end, state in segments:
        states[start:end] = state
    
    axes[1].plot(time_steps, states, drawstyle='steps-post', linewidth=2)
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Hidden State')
    axes[1].set_title('Hidden State Sequence')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="HMMR time-series segmentation")
    parser.add_argument("--predictions_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--wandb_project", type=str, default="weight-space-research")
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--num_states", type=str, default="5,10,15")
    parser.add_argument("--cluster_subsequences", action="store_true")
    parser.add_argument("--max_samples", type=int, default=1000)
    
    args = parser.parse_args()
    
    # Parse num_states
    num_states_list = [int(x) for x in args.num_states.split(',')]
    
    # Initialize WandB
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity if args.wandb_entity else None,
        name=f"hmmr_segmentation_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    print(f"\n{'='*80}")
    print(f"HMMR Time-Series Segmentation")
    print(f"{'='*80}\n")
    print(f"Number of states to try: {num_states_list}")
    print(f"Output directory: {args.output_dir}")
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Placeholder for actual implementation
    print(f"\n{'='*80}")
    print(f"HMMR segmentation module created.")
    print(f"Note: Requires predicted weight sequences for actual segmentation.")
    print(f"{'='*80}\n")
    
    wandb.finish()


if __name__ == "__main__":
    main()
