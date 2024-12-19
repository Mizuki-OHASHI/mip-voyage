from typing import Optional
from random import randint, seed, random


def make_input(
    box_num: int = 10,
    y_num: int = 3,
    seed_num: int = 0,
    name: Optional[str] = None,
    params: dict = {},
) -> dict:
    """Make input data for the MIP model.

    Args:
        box_num (int): Number of boxes
        y_num (int): Number of y
        seed_num (int): Seed number
        name (str): Name of the input data
        params (dict): Parameters for the input data
    ```
    params = {
        "y_num_rate": float,  # y_num is 1 with this rate (default: 0.8)
        "y_list_rate": float, # rate of selecting y_list (default: 0.5)
    }
    ```

    Returns:
        dict: Input data for the MIP model
    """

    seed(seed_num)
    if name is None:
        name = f"box_{box_num}_y_{y_num}_seed_{seed_num}"

    y_list = [
        {"id": i, "limit": -1} for i in range(y_num)
    ]  # TODO: limit to be random in the future

    z_num = randint(box_num // 4 + 1, box_num // 2 + 1)
    z_matrix = [[randint(2, 10) for _ in range(z_num)] for _ in range(z_num)]

    x_min_list = [randint(0, 20) for _ in range(box_num)]
    x_max_list = [x_min + randint(0, 40) for x_min in x_min_list]
    boxes = [
        {
            "id": i,
            "x_min": x_min_list[i],
            "x_max": x_max_list[i],
            "y_list": [
                y for y in range(y_num) if random() > params.get("y_list_rate", 0.5)
            ],
            "y_num": 1 if random() > params.get("y_num_rate", 0.8) else 2,
            "z_id": randint(0, z_num - 1),
        }
        for i in range(box_num)
    ]

    data = {
        "boxes": boxes,
        "y_list": y_list,
        "z_matrix": z_matrix,
    }

    return {"name": name, "data": data}
