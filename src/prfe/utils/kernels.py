"""Exact kernel functions — useful for validating random feature approximations."""

import torch


def rbf_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """K_+(x,y) = exp(-‖x-y‖² / (2σ²))  —  (N,d) × (M,d) → (N,M)"""
    diff = x.unsqueeze(1) - y.unsqueeze(0)             # (N, M, d)
    return torch.exp(-diff.pow(2).sum(-1) / (2 * sigma**2))


def polynomial_kernel(
    x: torch.Tensor, y: torch.Tensor, kappa: int, c: float = 0.0
) -> torch.Tensor:
    """K_-(x,y) = (xᵀy + c)^κ  —  (N,d) × (M,d) → (N,M)"""
    return (x @ y.T + c).pow(kappa)


def pontryagin_kernel(
    x: torch.Tensor,
    y: torch.Tensor,
    kappa: int,
    sigma: float = 1.0,
    c: float = 0.0,
) -> torch.Tensor:
    """K_Pontryagin(x,y) = K_+(x,y) - K_-(x,y)  —  (N,d) × (M,d) → (N,M)"""
    return rbf_kernel(x, y, sigma) - polynomial_kernel(x, y, kappa, c)
