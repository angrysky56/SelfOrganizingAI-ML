"""
Tests for specific ethical scenarios and edge cases
"""

import pytest
import numpy as np
from datetime import datetime
from src.ethics.core.analyzers.bias_detector import BiasDetector
from src.ethics.core.analyzers.fairness_metrics import FairnessMetrics
from src.ethics.core.analyzers.simulation_guard import SimulationGuard

@pytest.fixture
def extreme_bias_data():
    """Data with clear demographic bias"""
    return {
        'predictions': np.array([1, 1, 1, 0, 0, 0, 0, 0]),
        'ground_truth': np.array([1, 1, 0, 1, 1, 0, 0, 0]),
        'protected_attributes': {
            'group': np.array(['A', 'A', 'A', 'B', 'B', 'B', 'B', 'B'])
        },
        'features': np.random.rand(8, 5)
    }

@pytest.fixture
def unstable_simulation_data():
    """Simulation data showing unstable patterns"""
    return {
        'state': {
            'values': np.exp(np.linspace(0, 10, 100)),  # Exponential growth
            'gradients': np.random.exponential(size=100)
        },
        'metrics': {
            'iterations': 500,
            'error': 2.5,
            'error_variance': 1.8
        },
        'resources': {
            'memory_usage': 0.95,  # Near resource limit
            'cpu_usage': 0.88
        }
    }

@pytest.fixture
def feedback_loop_data():
    """Data exhibiting potential feedback loop issues"""
    return {
        'predictions': np.array([1] * 40 + [0] * 60),  # Strongly skewed
        'features': np.vstack([
            np.random.normal(1, 0.1, (40, 5)),  # Cluster 1
            np.random.normal(-1, 0.1, (60, 5))  # Cluster 2
        ]),
        'iteration_history': [
            {'bias_score': score} for score in np.linspace(0.5, 0.9, 10)
        ]
    }

@pytest.mark.asyncio
async def test_extreme_bias_detection(extreme_bias_data):
    detector = BiasDetector()
    result = await detector.analyze(extreme_bias_data)
    
    # Should detect strong bias
    assert result.value < 0.3
    assert 'demographic' in result.details
    assert result.context.metadata['bias_breakdown']['demographic'] < 0.3

@pytest.mark.asyncio
async def test_unstable_simulation_detection(unstable_simulation_data):
    guard = SimulationGuard()
    result = await guard.analyze(unstable_simulation_data)
    
    # Should detect instability and resource issues
    assert result.value < 0.5
    assert any('resource' in v for v in result.details['violations'])
    assert any('stability' in v for v in result.details['violations'])

@pytest.mark.asyncio
async def test_feedback_loop_detection(feedback_loop_data):
    detector = BiasDetector()
    result = await detector.analyze(feedback_loop_data)
    
    # Should detect feedback loop patterns
    assert 'feedback_loop_risk' in result.details
    assert result.context.metadata.get('warnings', [])

@pytest.mark.asyncio
async def test_intersectional_fairness():
    """Test fairness across multiple intersecting protected attributes"""
    data = {
        'predictions': np.random.randint(0, 2, 1000),
        'protected_attributes': {
            'gender': np.random.choice(['M', 'F'], 1000),
            'age_group': np.random.choice(['young', 'middle', 'senior'], 1000),
            'ethnicity': np.random.choice(['A', 'B', 'C'], 1000)
        },
        'features': np.random.rand(1000, 10)
    }
    
    metrics = FairnessMetrics()
    result = await metrics.analyze(data)
    
    # Check intersectional metrics
    assert 'intersectional_disparity' in result.details
    assert isinstance(result.details['intersectional_disparity'], dict)

@pytest.mark.asyncio
async def test_temporal_stability():
    """Test stability over time with changing patterns"""
    timestamps = np.arange(100)
    states = []
    
    # Generate evolving patterns
    for t in timestamps:
        state = {
            'values': np.sin(t/10) * np.random.rand(50) + t/100,
            'iteration': t
        }
        states.append(state)
    
    guard = SimulationGuard()
    violations_detected = []
    
    # Analyze stability over time
    for state in states:
        result = await guard.analyze({'state': state})
        violations_detected.append(bool(result.details['violations']))
    
    # Check for pattern detection
    assert any(violations_detected), "Should detect some stability issues"
    assert not all(violations_detected), "Should not flag all states as problematic"
