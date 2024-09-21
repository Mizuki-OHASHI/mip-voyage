from datetime import datetime
from pprint import pprint
from typing import Tuple

from mip import Model, xsum, minimize, MINIMIZE, CBC, OptimizationStatus

from data import build_validate_data, read_file, parse_json


class MipModel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, sense=MINIMIZE, solver_name=CBC, **kwargs)

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
        # B[b1][b2] = Relu(Y[b1] + z_matrix(b1, b2) - Y[b2]) if b2 is above b1 else don't care
        self.B = [
            [
                self.add_var(
                    f"B_{b1}_{b2}",
                    var_type="C",
                    lb=0,
                )
                for b2 in range(self.box_num)
            ]
            for b1 in range(self.box_num)
        ]
        self.big_M_B = 100  # TODO: find a better value

    def model_input_validator(self):
        structure = {
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
        return build_validate_data(structure)

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
                self += self.Y[b1] + self.get_z(b1, b2) - self.Y[b2] <= self.B[b1][
                    b2
                ] + self.big_M_B * (1 - xsum(self.A[b1][b2]))

    def add_objective(self):
        self.objective = minimize(
            -10
            * xsum(
                self.X[b][y] for b in range(self.box_num) for y in range(len(self.X[b]))
            )
            + xsum(
                self.get_z(b1, b2) * self.A[b1][b2][y]
                for b1 in range(self.box_num)
                for b2 in range(self.box_num)
                for y in range(len(self.A_domain[b1][b2]))
            )
            + 10
            * xsum(
                self.B[b1][b2]
                for b1 in range(self.box_num)
                for b2 in range(self.box_num)
            )
            + xsum(self.Y)
        )

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
                [self.B[b1][b2].x for b2 in range(self.box_num)]
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
            result[y] = "   " * int(max_x)
        for box_id, x in boxes:
            x = int(x)
            result[y] = result[y][: 3 * x] + f" {box_id:02}" + result[y][3 * x + 3 :]
    for y in sorted(result.keys()):
        print(f"{y:02}|{result[y]}")


def run_mip_model(input_str: str) -> Tuple[bool, dict]:
    model = MipModel()
    validator = model.model_input_validator()
    ok, input = parse_json(input_str, validator)
    if not ok:
        return False, input  # error message
    model.set_input(input)
    model.add_constraints()
    model.add_objective()
    model.optimize(max_seconds_same_incumbent=10, max_seconds=60)
    if (
        model.status == OptimizationStatus.OPTIMAL
        or model.status == OptimizationStatus.FEASIBLE
    ):
        return True, model.decode()
    return False, {"message": "No optimal solution found", "status": model.status.value}


if __name__ == "__main__":
    log = lambda msg: print(f"{datetime.now().isoformat()}: {msg}")
    log("MIP model")
    model = MipModel()
    log("read input")
    SMALL_INPUT_PATH = "../asset/mip_small_input.json"
    MIDDLE_INPUT_PATH = "../asset/mip_middle_input.json"
    input_str = read_file(MIDDLE_INPUT_PATH)
    log("build validator")
    validator = model.model_input_validator()
    log("parse input")
    ok, input = parse_json(input_str, validator)
    if not ok:
        raise ValueError(str(input))

    pprint(input)
    log("set input")
    model.set_input(input)
    log("add constraints")
    model.add_constraints()
    log("add objective")
    model.add_objective()
    log("solve")
    model.optimize(max_seconds_same_incumbent=10, max_seconds=60)
    log("solution")
    print("Objective value:", model.objective_value)
    optimal = model.decode()
    pprint(optimal)
    visualize_optimal(optimal)
    print(model.decode_helpers())
