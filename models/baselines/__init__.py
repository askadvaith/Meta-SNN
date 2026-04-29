"""
Baseline Models for Comparison with Meta-Learning Approaches

This module contains standard (non-meta-learning) SNN models
for establishing baseline performance on olfactory datasets.
"""

from .baseline_snn import BaselineSNN, create_baseline_snn

__all__ = ['BaselineSNN', 'create_baseline_snn']
