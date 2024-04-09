#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 11:33:01 2024

@author: schoelleh96
"""
from GeoCS import Traj, Dist, Bound
from datetime import datetime
import pytest


# %% Basic Pytest

def test_type():
    T = Traj(".", datetime(2016, 5, 2, 0))
    assert isinstance(T, Traj)
