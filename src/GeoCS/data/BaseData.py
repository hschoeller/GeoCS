#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 10:43:18 2024

@author: schoelleh96
"""

from abc import ABC, abstractmethod
from datetime import datetime

class Data(ABC):
    """Abstract base class for all kinds of data in this package."""
    loadDataPath: str
    saveDataPath: str
    startDate: datetime
    def __init__(self, loadDataPath: str, saveDataPath: str,
                 startDate: datetime):
        self.loadDataPath = loadDataPath
        self.saveDataPath = saveDataPath
        if not isinstance(startDate, datetime):
            raise TypeError("startDate must be a datetime object")
        self.startDate = startDate

    def __str__(self) -> str:
        dateStr: str = self.startDate.strftime("%Y-%m-%d %H:%M:%S")
        return (f"Load from: {self.loadDataPath}, Save to: " +
                f"{self.saveDataPath}, Date: {dateStr}")

    @abstractmethod
    def load(self) -> None:
        """Load data from data_path. Implementation required."""
        pass

    @abstractmethod
    def save(self) -> None:
        """Save data to data_path. Implementation required."""
        pass

    @abstractmethod
    def plot(self) -> None:
        """Plot data. Implementation required."""
        pass
