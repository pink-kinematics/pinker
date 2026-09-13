# SPDX-License-Identifier: Apache-2.0

"""Utility classes and functions."""

import numpy as np


class VectorSpace:
    """Wrapper to refer to a vector space and its characteristic matrices."""

    __eye: np.ndarray
    __ones: np.ndarray
    __zeros: np.ndarray

    def __init__(self, dim: int):
        """Create new vector space description.

        Args:
            dim: Dimension.
        """
        eye = np.eye(dim)
        ones = np.ones(dim)
        zeros = np.zeros(dim)
        eye.setflags(write=False)
        ones.setflags(write=False)
        zeros.setflags(write=False)
        self.__eye = eye
        self.__ones = ones
        self.__zeros = zeros

    @property
    def eye(self) -> np.ndarray:
        """Identity matrix from and to the vector space."""
        return self.__eye

    @property
    def ones(self) -> np.ndarray:
        """Vector full of ones, dimension of the space."""
        return self.__ones

    @property
    def zeros(self) -> np.ndarray:
        """Zero vector of the space."""
        return self.__zeros
