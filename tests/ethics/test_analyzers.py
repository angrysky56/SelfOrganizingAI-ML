"""
Tests for ethics analyzers
"""

import pytest
import numpy as np
from datetime import datetime
from src.ethics.core.analyzers.bias_detector import BiasDetector
from src.ethics.core.analyzers.fairness_metrics import FairnessMetrics
from src.ethics.core.analyzers.simulation_guard import SimulationGuard

@pytest.fixture
def sample_data():
    return {
        'predictions': np.array([0, 1, 1, 0, 1]),
        'ground_truth': np.array([0, 1, 1, 0, 0]),
        'protected_attributes': {
            'gender': np.array(['M', 'F', 'M', 'F', 'M']),
            'age': np.array([25, 35, 45, 28, 50])
        },
        'features': np.random.rand(5, 10),
        'state': {
            'values': np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        },
        'metrics': {
            'iterations': 100,
            'error': 0.05
        },
        'resources': {
            'memory_usage': 0.5,
            'cpu_usage': 0.6
        }
    }

@pytest.mark.asyncio
async def test_bias_detector(sample_data):
    detector = BiasDetector()
    result = await detector.analyze(sample_data)
    assert 0 <= result.value <= 1
    assert 0 <= result.confidence <= 1
    assert result.context is not None

@pytest.mark.asyncio
async def test_fairness_metrics(sample_data):
    metrics = FairnessMetrics()
    result = await metrics.analyze(sample_data)
    assert 0 <= result.value <= 1
    assert 0 <= result.confidence <= 1
    assert 'demographic_parity' in result.details

@pytest.mark.asyncio
async def test_simulation_guard(sample_data):
    guard = SimulationGuard()
    result = await guard.analyze(sample_data)
    assert 0 <= result.value <= 1
    assert 0 <= result.confidence <= 1
    assert 'violations' in result.details

@pytest.mark.asyncio
async def test_guard_boundary_conditions(sample_data):
    guard = SimulationGuard()
    # Test normal boundaries
    result = await guard.analyze(sample_data)
    assert not result.details['violations']
    
    # Test violation
    sample_data['state']['values'] = np.array([1e6, 2e6])
    result = await guard.analyze(sample_data)
    assert result.details['violations']
