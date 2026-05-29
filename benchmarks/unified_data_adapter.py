from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd


@dataclass
class UnifiedTask:
    name: str
    source: str
    dim: int
    profile: str
    function_str: str
    var_data: dict
    y_obs: np.ndarray
    suite: str = ""


def _infer_profile(dim: int, dataset_name: str) -> str:
    n = dataset_name.lower()
    if dim == 2:
        return "multi"
    if any(k in n for k in ("sin", "cos", "trig", "periodic")):
        return "trig"
    if any(k in n for k in ("log",)):
        return "log"
    if any(k in n for k in ("sqrt", "root")):
        return "sqrt"
    return "poly"


def _resample_1d(x: np.ndarray, y: np.ndarray, n_points: int = 127) -> tuple[dict, np.ndarray]:
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    x_new = np.linspace(float(np.min(x)), float(np.max(x)), n_points)
    y_new = np.interp(x_new, x, y)
    return {"x": x_new}, y_new


def _resample_2d(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    nx: int = 32,
    ny: int = 32,
) -> tuple[dict, np.ndarray]:
    # 用最近邻映射到规则网格，避免引入 scipy 依赖
    xg = np.linspace(float(np.min(x)), float(np.max(x)), nx)
    yg = np.linspace(float(np.min(y)), float(np.max(y)), ny)
    xx, yy = np.meshgrid(xg, yg, indexing="xy")
    xx_flat, yy_flat = xx.ravel(), yy.ravel()

    pts = np.column_stack([x, y]).astype(np.float64, copy=False)
    grid = np.column_stack([xx_flat, yy_flat]).astype(np.float64, copy=False)
    # 分块避免一次性矩阵过大
    out = np.empty(len(grid), dtype=np.float64)
    chunk = 1024
    for i in range(0, len(grid), chunk):
        g = grid[i : i + chunk]
        d2 = (
            (g[:, None, 0] - pts[None, :, 0]) ** 2
            + (g[:, None, 1] - pts[None, :, 1]) ** 2
        )
        nn = np.argmin(d2, axis=1)
        out[i : i + chunk] = z[nn]

    return {"x": xx_flat, "y": yy_flat}, out


def tabular_to_unified_task(
    name: str,
    source: str,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Optional[List[str]] = None,
) -> UnifiedTask:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if X.ndim != 2:
        raise ValueError(f"{name}: X 必须是二维数组")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"{name}: X/y 样本数不一致")
    if X.shape[1] < 1:
        raise ValueError(f"{name}: 至少需要 1 个特征")

    # 当前 SRO 资产只支持 1D(127) 或 2D(32x32)
    if X.shape[1] == 1:
        var_data, y_obs = _resample_1d(X[:, 0], y, n_points=127)
        dim = 1
    else:
        var_data, y_obs = _resample_2d(X[:, 0], X[:, 1], y, nx=32, ny=32)
        dim = 2

    fn = feature_names or [f"x{i+1}" for i in range(X.shape[1])]
    fdesc = ", ".join(fn[:2]) if dim == 2 else fn[0]
    suite = "feynman" if source.startswith("feynman") else ("pmlb" if source == "pmlb" else "default")
    return UnifiedTask(
        name=name,
        source=source,
        dim=dim,
        profile=_infer_profile(dim, name),
        function_str=f"tabular target | features={fdesc}",
        var_data=var_data,
        y_obs=y_obs,
        suite=suite,
    )


def _require_pmlb():
    try:
        from pmlb import dataset_names, fetch_data  # noqa: F401
    except Exception as e:
        raise ImportError(
            "未检测到 pmlb。请先安装: pip install pmlb pandas"
        ) from e


def load_pmlb_tasks(dataset_list: Iterable[str]) -> List[UnifiedTask]:
    _require_pmlb()
    from pmlb import fetch_data

    tasks: List[UnifiedTask] = []
    for ds in dataset_list:
        data = fetch_data(ds, return_X_y=False, local_cache_dir="./.pmlb_cache")
        if "target" not in data.columns:
            continue
        num_df = data.select_dtypes(include=[np.number]).dropna()
        if "target" not in num_df.columns:
            continue
        feat_cols = [c for c in num_df.columns if c != "target"]
        if not feat_cols:
            continue
        use_cols = feat_cols[:2]
        X = num_df[use_cols].to_numpy(dtype=np.float64)
        y = num_df["target"].to_numpy(dtype=np.float64)
        tasks.append(tabular_to_unified_task(ds, "pmlb", X, y, feature_names=use_cols))
    return tasks


def discover_srbench_like_datasets(max_count: int = 20) -> List[str]:
    _require_pmlb()
    from pmlb import dataset_names

    keys = (
        "feynman",
        "keijzer",
        "korns",
        "nguyen",
        "pagie",
        "vladislavleva",
    )
    cand = [d for d in dataset_names if any(k in d.lower() for k in keys)]
    if cand:
        return sorted(cand)[:max_count]
    # 兜底：使用一组常见回归数据集名（按可用性取交集）
    fallback = [
        "1027_ESL",
        "1028_SWD",
        "1029_LEV",
        "1030_ERA",
        "1096_FacultySalaries",
        "192_vineyard",
        "197_cpu_act",
        "228_elusage",
        "229_pwLinear",
        "230_machine_cpu",
        "485_analcatdata_vehicle",
        "537_houses",
    ]
    avail = [d for d in fallback if d in dataset_names]
    return avail[:max_count]


def load_srbench_tasks(max_count: int = 20) -> List[UnifiedTask]:

    names = discover_srbench_like_datasets(
        max_count=max_count
    )

    tasks = load_pmlb_tasks(names) if names else []

    from pmlb import dataset_names, fetch_data

    cache_dir = "./.pmlb_cache"

    seen = {t.name for t in tasks}
    auto = []

    for ds in dataset_names:

        if len(tasks) + len(auto) >= max_count:
            break

        if ds in seen:
            continue

        # 没有缓存就跳过
        dataset_dir = os.path.join(cache_dir, ds)
        if not os.path.exists(dataset_dir):
            continue

        try:
            print(f"Loading local dataset: {ds}")

            data = fetch_data(
                ds,
                return_X_y=False,
                local_cache_dir=cache_dir
            )

            if "target" not in data.columns:
                continue

            num_df = (
                data.select_dtypes(include=[np.number])
                .dropna()
            )

            if "target" not in num_df.columns:
                continue

            y = num_df["target"].to_numpy(
                dtype=np.float64
            )

            if np.unique(y).size < 20:
                continue

            feat_cols = [
                c for c in num_df.columns
                if c != "target"
            ]

            if not feat_cols:
                continue

            X = num_df[feat_cols[:2]].to_numpy(
                dtype=np.float64
            )

            auto.append(
                tabular_to_unified_task(
                    ds,
                    "srbench",
                    X,
                    y,
                    feature_names=feat_cols[:2],
                )
            )

        except Exception as e:
            print(f"skip {ds}: {e}")

    return (tasks + auto)[:max_count]


def load_feynman_sr_tasks(max_count: int = 20) -> List[UnifiedTask]:
    _require_pmlb()
    from pmlb import dataset_names

    names = [d for d in dataset_names if "feynman" in d.lower()]
    names = sorted(names)[:max_count]
    tasks = load_pmlb_tasks(names) if names else []
    if tasks:
        return tasks

    # 若 pmlb 无 feynman 数据，则使用内置 Feynman 公式生成数据
    rng = np.random.default_rng(42)
    builtins = [
        ("Feynman-I.6.2a", 1, lambda x: np.exp(-x**2 / 2.0), (-3.0, 3.0)),
        ("Feynman-I.10.7", 1, lambda x: x * np.sin(x), (-3.0, 3.0)),
        ("Feynman-I.12.1", 1, lambda x: np.sin(2.0 * x), (-np.pi, np.pi)),
        ("Feynman-I.13.4", 1, lambda x: np.exp(-x) * np.sin(x), (0.0, 5.0)),
        ("Feynman-I.18.4", 2, lambda x, y: x * y, (-2.0, 2.0)),
        ("Feynman-I.30.5", 2, lambda x, y: np.sin(x) * np.cos(y), (-np.pi, np.pi)),
        ("Feynman-II.2.42", 2, lambda x, y: np.sqrt(np.abs(x)) + np.log(np.abs(y) + 1.0), (0.0, 4.0)),
        ("Feynman-III.4.32", 2, lambda x, y: np.power(np.abs(x) + 1e-8, y), (0.2, 1.5)),
    ]
    tasks_builtin: List[UnifiedTask] = []
    for name, dim, fn, xr in builtins[:max_count]:
        if dim == 1:
            x = rng.uniform(xr[0], xr[1], size=2000)
            y = fn(x)
            X = x.reshape(-1, 1)
            tasks_builtin.append(
                tabular_to_unified_task(name, "feynman-sr", X, y, feature_names=["x"])
            )
        else:
            x = rng.uniform(xr[0], xr[1], size=4000)
            y = rng.uniform(xr[0], xr[1], size=4000)
            z = fn(x, y)
            X = np.column_stack([x, y])
            tasks_builtin.append(
                tabular_to_unified_task(name, "feynman-sr", X, z, feature_names=["x", "y"])
            )
    return tasks_builtin[:max_count]

