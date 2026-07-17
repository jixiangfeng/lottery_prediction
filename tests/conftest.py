# -*- coding: utf-8 -*-
import random

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def set_random_seed() -> None:
    """
    固定随机种子，确保测试结果稳定。
    """

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
