from quantlab.returns import calculate_portfolio_return
from quantlab.returns import calculate_portfolio_return_general
from quantlab.returns import calculate_portfolio_return_numpy
from quantlab.returns import calculate_volatility
from quantlab.returns import calculate_correlation
import pytest

def test_calculate_portfolio_return():
    result = calculate_portfolio_return(0.10, 0.25, 0.04, 0.75)

    assert result == 0.055

def test_calculate_portfolio_return_when_both_assets_return_zero():
    result = calculate_portfolio_return(0.0, 0.5, 0.0, 0.5)

    assert result == 0.0

def test_calculate_portfolio_return_with_a_loss():
    result = calculate_portfolio_return(-0.10, 0.5, 0.04, 0.5)

    assert result == pytest.approx(-0.03)

def test_calculate_portfolio_return_general_with_three_assets():
    returns = [0.10, 0.04, 0.08]
    weights = [0.25, 0.50, 0.25]

    result = calculate_portfolio_return_general(returns, weights)

    assert result == pytest.approx(0.065)

def test_calculate_portfolio_return_general_rejects_different_lengths():
    returns = [0.10, 0.04, 0.08]
    weights = [0.50, 0.50]

    with pytest.raises(ValueError, match="same length"):
        calculate_portfolio_return_general(returns, weights)

def test_calculate_portfolio_return_numpy_with_three_assets():
    returns = [0.10, 0.04, 0.08]
    weights = [0.25, 0.50, 0.25]

    result = calculate_portfolio_return_numpy(returns, weights)

    assert result == pytest.approx(0.065)

def test_calculate_portfolio_return_numpy_rejects_different_lengths():
    returns = [0.10, 0.04, 0.08]
    weights = [0.50, 0.50]

    with pytest.raises(ValueError, match="same length"):
        calculate_portfolio_return_numpy(returns, weights)

def test_calculate_portfolio_return_functions_consistency():
    returns = [0.10, 0.04, 0.08]
    weights = [0.25, 0.50, 0.25]

    result_general = calculate_portfolio_return_general(returns, weights)
    result_numpy = calculate_portfolio_return_numpy(returns, weights)

    assert result_general == pytest.approx(result_numpy)

def test_calculate_volatility():
    returns = [0.02, -0.01, 0.03, -0.02, 0.01]

    result = calculate_volatility(returns)

    assert result == pytest.approx(0.01854723699)

def test_calculate_volatility_with_constant_returns():
    returns = [0.05, 0.05, 0.05, 0.05]

    result = calculate_volatility(returns)

    assert result == 0.0

def test_calculate_correlation_negative_one():
    returns_a = [0.01, 0.02, 0.03, 0.04, 0.05]
    returns_b = [0.05, 0.04, 0.03, 0.02, 0.01]

    result = calculate_correlation(returns_a, returns_b)

    assert result == pytest.approx(-1.0)

def test_calculate_correlation_rejects_different_lengths():
    returns_a = [0.01, 0.02, 0.03, 0.04, 0.05]
    returns_b = [0.05, 0.04, 0.03, 0.02]

    with pytest.raises(ValueError, match="same length"):
        calculate_correlation(returns_a, returns_b)