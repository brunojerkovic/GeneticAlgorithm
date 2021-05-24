from dataclasses import dataclass, field

@dataclass(
    init=True,
    repr=True,
    eq=True,
    order=False
)
class Problem:
    cities: list = field(default_factory=list)
    items: list = field(default_factory=list)
    knapsack_dimension: list = field(default_factory=list)
    problem_name: str = ''
    map_dimension: tuple = (0, 0)
    item_dimension: tuple = (0, 0)
    kp_capacity: int = 0
    min_speed: float = 0.
    max_speed: float = 0.
    renting_ratio: float = 0.
    total_profit: float = 0.