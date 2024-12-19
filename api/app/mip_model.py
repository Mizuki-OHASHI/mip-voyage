from datetime import datetime
from pprint import pprint
from typing import Tuple
from os import path
import sys

from mip import Model, xsum, minimize, MINIMIZE, CBC, OptimizationStatus

from make_input import make_input
from data import build_validate_data, read_file, parse_json


class MipModel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, sense=MINIMIZE, solver_name=CBC, **kwargs)
        self.input_structure = {
            "name": str,
            "data": {
                "boxes": [
                    {
                        "id": int,
                        "x_min": float,
                        "x_max": float,
                        "y_list": [int],
                        "y_num": int,
                        "z_id": int,
                    }
                ],
                "y_list": [{"id": int, "limit": int}],
                "z_matrix": [[float]],
            },
        }

    def set_input(self, input):
        """Set input data to the model.

        Args:
            input (dict): JSON data, assumed to be validated by self.model_input_validator
        """
        name = input["name"]
        data = input["data"]
        self.instance_name = name
        self.data = data

        def get_z(b1: int, b2: int) -> float:
            b1_z = data["boxes"][b1]["z_id"]
            b2_z = data["boxes"][b2]["z_id"]
            return data["z_matrix"][b1_z][b2_z]

        self.get_z = get_z

        self.box_num = len(data["boxes"])

        # X[b][y] = 1 if box b is at self.data["y_list"][y]
        self.X = [
            [
                self.add_var(f"X_{b}_{y}", var_type="B")
                for y in self.data["boxes"][b]["y_list"]
            ]
            for b in range(self.box_num)
        ]

        # help X
        # A[b1][b2][y] = 1 if box b2 is above box b1 in terms of value x, and both are at y
        self.A_domain = [
            [
                (
                    []
                    if b1 == b2
                    else [
                        y
                        for y in self.data["boxes"][b1]["y_list"]
                        if y in self.data["boxes"][b2]["y_list"]
                    ]
                )
                for b2 in range(self.box_num)
            ]
            for b1 in range(self.box_num)
        ]
        self.A = [
            [
                [
                    self.add_var(f"A_{b1}_{b2}_{y}", var_type="B")
                    for y in self.A_domain[b1][b2]
                ]
                for b2 in range(self.box_num)
            ]
            for b1 in range(self.box_num)
        ]

        # help X
        # H[y][b] = 1 if box b is at y and is the head of the list
        # T[y][b] = 1 if box b is at y and is the tail of the list
        self.HT_domain = [
            [b for b in range(self.box_num) if y in self.data["boxes"][b]["y_list"]]
            for y in range(len(data["y_list"]))
        ]
        self.H = [
            [self.add_var(f"H_{y}_{b}", var_type="B") for b in self.HT_domain[y]]
            for y in range(len(data["y_list"]))
        ]
        self.T = [
            [self.add_var(f"T_{y}_{b}", var_type="B") for b in self.HT_domain[y]]
            for y in range(len(data["y_list"]))
        ]

        # Y[b] = x value of box b
        self.Y = [
            self.add_var(name=f"Y_{i}", var_type="C", lb=box["x_min"], ub=box["x_max"])
            for i, box in enumerate(data["boxes"])
        ]

        # help Y
        # B[b1][b2] = ReLU(Y[b1] + z_matrix(b1, b2) - Y[b2]) if b2 is above b1 else don't care
        self.B = [
            [
                self.add_var(
                    f"B_{b1}_{b2}",
                    var_type="B",
                    lb=0,
                )
                for b2 in range(self.box_num)
            ]
            for b1 in range(self.box_num)
        ]
        self.big_M_B = 100  # TODO: find a better value

    @property
    def model_input_validator(self):
        return build_validate_data(self.input_structure)

    def add_constraints(self):
        # constraint on X
        for i, box in enumerate(self.data["boxes"]):
            self += xsum(self.X[i]) <= box["y_num"]

        # constraint to bind A, H, T with X
        for y in range(len(self.data["y_list"])):
            for b1 in range(self.box_num):
                if y not in self.data["boxes"][b1]["y_list"]:
                    continue
                y_idx = self.data["boxes"][b1]["y_list"].index(y)
                b1_idx = self.HT_domain[y].index(b1)
                self += self.X[b1][y_idx] - self.T[y][b1_idx] == xsum(
                    self.A[b1][b2][self.A_domain[b1][b2].index(y)]
                    for b2 in range(self.box_num)
                    if y in self.A_domain[b1][b2]
                )
            for b2 in range(self.box_num):
                if y not in self.data["boxes"][b2]["y_list"]:
                    continue
                y_idx = self.data["boxes"][b2]["y_list"].index(y)
                b2_idx = self.HT_domain[y].index(b2)
                self += self.X[b2][y_idx] - self.H[y][b2_idx] == xsum(
                    self.A[b1][b2][self.A_domain[b1][b2].index(y)]
                    for b1 in range(self.box_num)
                    if y in self.A_domain[b1][b2]
                )

        # constraint on H and T
        for y in range(len(self.data["y_list"])):
            self += xsum(self.H[y]) == 1
            self += xsum(self.T[y]) == 1

        # constraint to bind B with Y (and A)
        for b1 in range(self.box_num):
            for b2 in range(self.box_num):
                self += self.Y[b1] + self.get_z(b1, b2) - self.Y[b2] <= self.big_M_B * (
                    self.B[b1][b2] + 1 - xsum(self.A[b1][b2])
                )

    def add_objective(self):
        self.objective = minimize(
            100
            * (
                xsum(box["y_num"] for box in self.data["boxes"])
                - xsum(
                    self.X[b][y]
                    for b in range(self.box_num)
                    for y in range(len(self.X[b]))
                )
            )
            + xsum(
                self.get_z(b1, b2) * self.A[b1][b2][y]
                for b1 in range(self.box_num)
                for b2 in range(self.box_num)
                for y in range(len(self.A_domain[b1][b2]))
            )
            + 10000
            * xsum(
                self.B[b1][b2]
                for b1 in range(self.box_num)
                for b2 in range(self.box_num)
            )
            + xsum(self.Y)
        )

    @property
    def detailed_objective_value(self):
        if self.objective_value is None:
            return None
        return {
            "sum_X": sum(box["y_num"] for box in self.data["boxes"])
            - sum(
                self.X[b][y].x
                for b in range(self.box_num)
                for y in range(len(self.X[b]))
            ),
            "sum_A": sum(
                self.A[b1][b2][y].x
                for b1 in range(self.box_num)
                for b2 in range(self.box_num)
                for y in range(len(self.A_domain[b1][b2]))
            ),
            "sum_B": sum(
                self.B[b1][b2].x
                for b1 in range(self.box_num)
                for b2 in range(self.box_num)
            ),
            "sum_Y": sum(self.Y[b].x for b in range(self.box_num)),
            "objective": self.objective_value,
        }

    def decode(self) -> dict:
        return {
            "boxes": [
                {
                    "id": box["id"],
                    "x": self.Y[i].x,
                    "y": [
                        self.data["boxes"][i]["y_list"][y]
                        for y, x in enumerate(self.X[i])
                        if x.x > 0
                    ],
                }
                for i, box in enumerate(self.data["boxes"])
            ]
        }

    def decode_helpers(self) -> dict:
        return {
            "A": [
                [
                    [
                        int(self.A[b1][b2][y].x)
                        for y in range(len(self.A_domain[b1][b2]))
                    ]
                    for b2 in range(self.box_num)
                ]
                for b1 in range(self.box_num)
            ],
            "H": [
                [int(self.H[y][b].x) for b in range(len(self.H[y]))]
                for y in range(len(self.data["y_list"]))
            ],
            "T": [
                [int(self.T[y][b].x) for b in range(len(self.T[y]))]
                for y in range(len(self.data["y_list"]))
            ],
            "B": [
                [round(self.B[b1][b2].x) for b2 in range(self.box_num)]
                for b1 in range(self.box_num)
            ],
        }


def visualize_optimal(decoded: dict):
    """Visualize the optimal solution on console."""
    boxes = decoded["boxes"]
    boxes_by_y = {}
    for box in boxes:
        for y in box["y"]:
            if y not in boxes_by_y:
                boxes_by_y[y] = []
            boxes_by_y[y].append((box["id"], box["x"]))
    # visualize like a gannt chart
    max_x = max(box["x"] for box in boxes)
    result = {}
    for y, boxes in boxes_by_y.items():
        if y not in result:
            result[y] = " " * 3 * int(max_x)
        for box_id, x in boxes:
            x = int(x)
            result[y] = result[y][: 3 * x] + f" {box_id:2}" + result[y][3 * x + 3 :]
    for y in sorted(result.keys()):
        print(f"{y:2}|{result[y]}")


def run_mip_model(input_str: str) -> Tuple[bool, dict]:
    model = MipModel()
    ok, input = parse_json(input_str, model.model_input_validator)
    if not ok:
        return False, input  # error message
    model.set_input(input)
    model.add_constraints()
    model.add_objective()
    status = model.optimize(max_seconds_same_incumbent=10, max_seconds=60)
    if status == OptimizationStatus.OPTIMAL or status == OptimizationStatus.FEASIBLE:
        solution = model.decode()
        objective_value = model.objective_value
        return True, {
            "solution": solution,
            "status": status.name,
            "objective": objective_value,
        }
    return False, {"message": "No optimal solution found", "status": status.name}


if __name__ == "__main__":
    model = MipModel()

    BASE = "/usr/src/api/asset"
    SMALL_INPUT_PATH = path.join(BASE, "mip_small_input.json")
    MIDDLE_INPUT_PATH = path.join(BASE, "mip_middle_input.json")
    input_type = sys.argv[1] if len(sys.argv) > 1 else "small"
    if input_type in ["small", "middle"]:
        print(f"Running MIP model with {input_type} input")
        model_path = {"small": SMALL_INPUT_PATH, "middle": MIDDLE_INPUT_PATH}
        input_str = read_file(model_path[input_type])
        ok, input_model = parse_json(input_str, model.model_input_validator)
        if not ok:
            raise ValueError(str(input))
    elif input_type == "random":
        params_list = [  # [name, label, default]
            ["box_num", "box number", 10],
            ["y_num", "y number", 3],
            ["seed_num", "seed number", 0],
        ]
        params = {}

        for name, label, default in params_list:
            value = input(f"Enter {label} (default: {default}): ")
            try:
                value = int(value) if value else default
            except ValueError:
                print(f"Invalid value: {value}")
                print(f"Using default value: {default}")
                value = default
            params[name] = value

        input_model = make_input(**params)
        ok, info = model.model_input_validator(input_model)
        if not ok:
            raise ValueError(str(info))
    else:
        raise ValueError("Usage: python mip_model.py [small|middle|random]")

    pprint(input_model)
    model.set_input(input_model)
    model.add_constraints()
    model.add_objective()
    model.optimize(max_seconds_same_incumbent=10, max_seconds=120)
    print("Objective value:", model.objective_value)
    optimal = model.decode()
    pprint(optimal)
    visualize_optimal(optimal)
    helpers = model.decode_helpers()
    print("H:", helpers["H"], "T:", helpers["T"])
    positive_A = {}
    for b1 in range(model.box_num):
        for b2 in range(model.box_num):
            for y_idx in range(len(helpers["A"][b1][b2])):
                if helpers["A"][b1][b2][y_idx] > 0:
                    y = model.A_domain[b1][b2][y_idx]
                    if y not in positive_A:
                        positive_A[y] = []
                    positive_A[y].append((b1, b2))
    print("list of A with positive value:", positive_A)
    positive_B = []
    for b1 in range(model.box_num):
        for b2 in range(model.box_num):
            if helpers["B"][b1][b2] > 0:
                positive_B.append((b1, b2))
    print("list of B with positive value:", positive_B)
    pprint({"detailed_objective": model.detailed_objective_value})
