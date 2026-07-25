"""PDE definitions for the three reproduction cases."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Dict, Type

import deepxde as dde
import numpy as np
import torch

from .utils import REFERENCE_DIR


DEFAULT_NUM_DOMAIN_POINTS = 8192
DEFAULT_NUM_BOUNDARY_POINTS = 2048
DEFAULT_NUM_INITIAL_POINTS = 2048


def cache_tensor(func):
    cache = {}
    sentinel = object()

    @functools.wraps(func)
    def wrapper(tensorlike_arg):
        key = (
            *tensorlike_arg.shape,
            torch.sin(tensorlike_arg).sum().item(),
            torch.cosh(tensorlike_arg).sum().item(),
        )
        result = cache.get(key, sentinel)
        if result is sentinel:
            result = func(tensorlike_arg)
            cache[key] = result
        return result

    return wrapper


class BaseCase:
    case_id: int
    name: str
    train_distribution = "Hammersley"

    def __init__(self):
        self.pde = None
        self.bcs = []
        self.loss_config = []
        self.ref_sol = None
        self.ref_data = None
        self.num_domain_points = DEFAULT_NUM_DOMAIN_POINTS
        self.num_boundary_points = DEFAULT_NUM_BOUNDARY_POINTS
        self.num_initial_points = None
        self.num_test_points = 1
        # Match the original PINNacle BasePDE constructor, which resets NumPy
        # before DeepXDE samples the collocation points.
        np.random.seed(42)

    @property
    def input_dim(self) -> int:
        return self.geom.dim

    @property
    def output_dim(self) -> int:
        return len(self.output_config)

    @property
    def num_pde(self) -> int:
        return sum(item["type"] == "pde" for item in self.loss_config)

    def set_pdeloss(self, names=None, num: int = 1) -> None:
        if names is not None:
            self.loss_config += [{"name": name, "type": "pde"} for name in names]
        else:
            self.loss_config += [{"name": f"pde_{i}", "type": "pde"} for i in range(num)]

    def add_bcs(self, config, geom=None) -> None:
        geom = geom if geom is not None else self.geom
        for bc in config:
            if bc.get("name") is None:
                bc["name"] = bc["type"] + ("" if bc["type"] == "ic" else "bc") + f"_{len(self.bcs) + 1}"

            if bc["type"] == "dirichlet":
                self.bcs.append(dde.DirichletBC(geom, bc["function"], bc["bc"], component=bc["component"]))
            elif bc["type"] == "robin":
                self.bcs.append(dde.RobinBC(geom, bc["function"], bc["bc"], component=bc["component"]))
            elif bc["type"] == "ic":
                self.bcs.append(dde.IC(geom, bc["function"], bc["bc"], component=bc["component"]))
            else:
                raise ValueError(f"Unsupported BC type in this release: {bc['type']}")
            self.loss_config.append({"name": bc["name"], "type": "boundary"})

    def make_data(self, sample: bool = True):
        raise NotImplementedError


class Poisson2DManyArea(BaseCase):
    case_id = 5
    name = "Poisson2D_ManyArea"

    def __init__(
        self,
        datapath: Path | None = None,
        a_cof_path: Path | None = None,
        f_cof_path: Path | None = None,
        bbox=(-10.0, 10.0, -10.0, 10.0),
        split=(5, 5),
        freq: int = 2,
    ):
        super().__init__()
        self.bbox = tuple(float(v) for v in bbox)
        self.split = tuple(int(v) for v in split)
        self.freq = int(freq)
        self.output_config = [{"name": "u"}]
        self.geom = dde.geometry.Rectangle(
            xmin=[self.bbox[0], self.bbox[2]], xmax=[self.bbox[1], self.bbox[3]]
        )

        a_path = Path(a_cof_path) if a_cof_path is not None else REFERENCE_DIR / "poisson_a_coef.dat"
        f_path = Path(f_cof_path) if f_cof_path is not None else REFERENCE_DIR / "poisson_f_coef.dat"
        self.a_cof = np.loadtxt(a_path)
        self.f_cof = np.loadtxt(f_path).reshape(
            self.split[0], self.split[1], self.freq, self.freq
        )
        block_size = np.array(
            [
                (self.bbox[1] - self.bbox[0] + 2e-5) / self.split[0],
                (self.bbox[3] - self.bbox[2] + 2e-5) / self.split[1],
            ]
        )

        def domain(x):
            reduced_x = x - np.array(self.bbox[::2]) + 1e-5
            dom = np.floor(reduced_x / block_size).astype("int32")
            return dom, reduced_x - dom * block_size

        def a(x):
            dom, _ = domain(x)
            return self.a_cof[dom[0], dom[1]]

        a = np.vectorize(a, signature="(2)->()")

        def f(x):
            dom, res = domain(x)

            def f_fn(coef):
                ans = coef[0, 0]
                for i in range(coef.shape[0]):
                    for j in range(coef.shape[1]):
                        tmp = np.sin(np.pi * np.array((i, j)) * (res / block_size))
                        ans += coef[i, j] * tmp[0] * tmp[1]
                return ans

            return f_fn(self.f_cof[dom[0], dom[1]])

        f = np.vectorize(f, signature="(2)->()")

        @cache_tensor
        def get_coef(x):
            x_np = x.detach().cpu().numpy()
            a_values = a(x_np)
            f_values = f(x_np)
            a_tensor = torch.tensor(a_values, dtype=x.dtype, device=x.device).unsqueeze(-1)
            f_tensor = torch.tensor(f_values, dtype=x.dtype, device=x.device).unsqueeze(-1)
            return a_tensor, f_tensor

        def poisson_pde(x, u):
            u_xx = dde.grad.hessian(u, x, i=0, j=0)
            u_yy = dde.grad.hessian(u, x, i=1, j=1)
            a_values, f_values = get_coef(x)
            return a_values * (u_xx + u_yy) + f_values

        self.pde = poisson_pde
        self.set_pdeloss(num=1)
        self.add_bcs(
            [
                {
                    "component": 0,
                    "function": lambda _, y: -y,
                    "bc": lambda _, on_boundary: on_boundary,
                    "type": "robin",
                }
            ]
        )

        ref_path = Path(datapath) if datapath is not None else REFERENCE_DIR / "poisson_manyarea.dat"
        self.ref_data = np.loadtxt(ref_path, comments="%", encoding="utf-8").astype(np.float64)

    def make_data(self, sample: bool = True):
        num_domain = self.num_domain_points if sample else 0
        num_boundary = self.num_boundary_points if sample else 0
        return dde.data.PDE(
            self.geom,
            self.pde,
            self.bcs,
            num_domain=num_domain,
            num_boundary=num_boundary,
            train_distribution=self.train_distribution,
            num_test=self.num_test_points,
        )


class Heat2DMultiscale(BaseCase):
    case_id = 7
    name = "Heat2D_Multiscale"

    def __init__(
        self,
        bbox=(0.0, 1.0, 0.0, 1.0, 0.0, 5.0),
        pde_coef=(1.0 / np.square(500.0 * np.pi), 1.0 / np.square(np.pi)),
        init_coef=(20.0 * np.pi, np.pi),
    ):
        super().__init__()
        self.bbox = tuple(float(v) for v in bbox)
        self.pde_coef = pde_coef
        self.init_coef = init_coef
        self.output_config = [{"name": "u"}]
        self.geom = dde.geometry.Rectangle(
            xmin=[self.bbox[0], self.bbox[2]], xmax=[self.bbox[1], self.bbox[3]]
        )
        timedomain = dde.geometry.TimeDomain(self.bbox[4], self.bbox[5])
        self.geomtime = dde.geometry.GeometryXTime(self.geom, timedomain)
        self.num_initial_points = DEFAULT_NUM_INITIAL_POINTS

        def heat_pde(x, u):
            u_xx = dde.grad.hessian(u, x, i=0, j=0)
            u_yy = dde.grad.hessian(u, x, i=1, j=1)
            u_t = dde.grad.jacobian(u, x, j=2)
            return [u_t - pde_coef[0] * u_xx - pde_coef[1] * u_yy]

        def ref_sol(xt):
            return (
                np.sin(init_coef[0] * xt[:, 0:1])
                * np.sin(init_coef[1] * xt[:, 1:2])
                * np.exp(
                    -(pde_coef[0] * init_coef[0] ** 2 + pde_coef[1] * init_coef[1] ** 2)
                    * xt[:, 2:3]
                )
            )

        self.pde = heat_pde
        self.ref_sol = ref_sol
        self.set_pdeloss(num=1)
        self.add_bcs(
            [
                {
                    "component": 0,
                    "function": ref_sol,
                    "bc": lambda _, on_initial: on_initial,
                    "type": "ic",
                },
                {
                    "component": 0,
                    "function": lambda _: 0,
                    "bc": lambda _, on_boundary: on_boundary,
                    "type": "dirichlet",
                },
            ],
            geom=self.geomtime,
        )

    @property
    def input_dim(self) -> int:
        return self.geomtime.dim

    def make_data(self, sample: bool = True):
        num_domain = self.num_domain_points if sample else 0
        num_boundary = self.num_boundary_points if sample else 0
        num_initial = self.num_initial_points if sample else 0
        return dde.data.TimePDE(
            self.geomtime,
            self.pde,
            self.bcs,
            num_domain=num_domain,
            num_boundary=num_boundary,
            num_initial=num_initial,
            train_distribution=self.train_distribution,
            num_test=self.num_test_points,
        )


class NS2DBackStep(BaseCase):
    case_id = 11
    name = "NS2D_BackStep"

    def __init__(
        self,
        datapath: Path | None = None,
        nu: float = 1.0 / 100.0,
        bbox=(0.0, 4.0, 0.0, 2.0),
    ):
        super().__init__()
        self.bbox = tuple(float(v) for v in bbox)
        self.nu = float(nu)
        self.output_config = [{"name": name} for name in ("u", "v", "p")]
        eps = 1e-5
        self.geom = dde.geometry.Rectangle(
            xmin=[self.bbox[0], self.bbox[2]], xmax=[self.bbox[1], self.bbox[3]]
        )
        rec = dde.geometry.Rectangle(
            xmin=[self.bbox[0] - eps, self.bbox[3] / 2.0],
            xmax=[self.bbox[1] / 2.0, self.bbox[3] + eps],
        )
        self.geom = dde.geometry.csg.CSGDifference(self.geom, rec)

        def ns_pde(x, u):
            u_vel, v_vel = u[:, 0:1], u[:, 1:2]
            u_vel_x = dde.grad.jacobian(u, x, i=0, j=0)
            u_vel_y = dde.grad.jacobian(u, x, i=0, j=1)
            u_vel_xx = dde.grad.hessian(u, x, component=0, i=0, j=0)
            u_vel_yy = dde.grad.hessian(u, x, component=0, i=1, j=1)

            v_vel_x = dde.grad.jacobian(u, x, i=1, j=0)
            v_vel_y = dde.grad.jacobian(u, x, i=1, j=1)
            v_vel_xx = dde.grad.hessian(u, x, component=1, i=0, j=0)
            v_vel_yy = dde.grad.hessian(u, x, component=1, i=1, j=1)

            p_x = dde.grad.jacobian(u, x, i=2, j=0)
            p_y = dde.grad.jacobian(u, x, i=2, j=1)

            momentum_x = u_vel * u_vel_x + v_vel * u_vel_y + p_x - nu * (u_vel_xx + u_vel_yy)
            momentum_y = u_vel * v_vel_x + v_vel * v_vel_y + p_y - nu * (v_vel_xx + v_vel_yy)
            continuity = u_vel_x + v_vel_y
            return [momentum_x, momentum_y, continuity]

        def boundary_in(x, on_boundary):
            return on_boundary and np.isclose(x[0], self.bbox[0])

        def boundary_out(x, on_boundary):
            return on_boundary and np.isclose(x[0], self.bbox[1])

        def boundary_other(x, on_boundary):
            return on_boundary and not (boundary_in(x, on_boundary) or boundary_out(x, on_boundary))

        def u_func(x):
            return x[:, 1:2] * (1.0 - x[:, 1:2]) * 4.0

        self.pde = ns_pde
        self.set_pdeloss(names=["momentum_x", "momentum_y", "continuity"])
        self.add_bcs(
            [
                {"component": 0, "function": u_func, "bc": boundary_in, "type": "dirichlet"},
                {"component": 1, "function": lambda _: 0, "bc": boundary_in, "type": "dirichlet"},
                {"component": 2, "function": lambda _: 0, "bc": boundary_out, "type": "dirichlet"},
                {"component": 0, "function": lambda _: 0, "bc": boundary_other, "type": "dirichlet"},
                {"component": 1, "function": lambda _: 0, "bc": boundary_other, "type": "dirichlet"},
            ]
        )

        ref_path = Path(datapath) if datapath is not None else REFERENCE_DIR / "ns_0_obstacle.dat"
        self.ref_data = np.loadtxt(ref_path, comments="%", encoding="utf-8").astype(np.float64)

    def make_data(self, sample: bool = True):
        num_domain = self.num_domain_points if sample else 0
        num_boundary = self.num_boundary_points if sample else 0
        return dde.data.PDE(
            self.geom,
            self.pde,
            self.bcs,
            num_domain=num_domain,
            num_boundary=num_boundary,
            train_distribution=self.train_distribution,
            num_test=self.num_test_points,
        )


CASES: Dict[int, Type[BaseCase]] = {
    5: Poisson2DManyArea,
    7: Heat2DMultiscale,
    11: NS2DBackStep,
}


def create_case(case_id: int) -> BaseCase:
    if case_id not in CASES:
        raise ValueError(f"Unsupported case_id={case_id}. This release includes cases 5, 7, and 11.")
    return CASES[case_id]()
