from dataclasses import dataclass, field
import numpy as np

@dataclass(
    init=True,
    repr=True,
    eq=True,
    order=False
)
class Chromosome:
    path: list = field(default_factory=list)
    knapsack: list = field(default_factory=list) # Ima "problem.item_dimension[0]" elemenata
    fit_kp: float = 0.
    fit_tsp: float = 0.
    fit: float = 0.
    age: int = 0
