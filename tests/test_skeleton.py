# -*- coding: utf-8 -*-

import pytest
from foolbox_evaluation.skeleton import fib

__author__ = "Stefanie Stoppel"
__copyright__ = "Stefanie Stoppel"
__license__ = "mit"


def test_fib():
    assert fib(1) == 1
    assert fib(2) == 1
    assert fib(7) == 13
    with pytest.raises(AssertionError):
        fib(-10)
