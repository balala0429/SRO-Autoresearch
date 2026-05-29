import numpy as np

# 扩展符号回归基准（用于 Nguyen 之外的实验）
# 字段与 train.py 中 NGUYEN_BENCHMARKS 保持一致
EXTRA_BENCHMARKS = [
    {
        "name": "Koza-2",
        "function_str": "(x^5 - 2*x^3 + x)",
        "dim": 1,
        "suite": "extra",
        "source": "extra",
        "profile": "poly",
        "x_range": (-1.0, 1.0),
        "target": lambda x: x ** 5 - 2 * x ** 3 + x,
    },
    {
        "name": "Koza-3",
        "function_str": "(x^6 - 2*x^4 + x^2)",
        "dim": 1,
        "suite": "extra",
        "source": "extra",
        "profile": "poly",
        "x_range": (-1.0, 1.0),
        "target": lambda x: x ** 6 - 2 * x ** 4 + x ** 2,
    },
    {
        "name": "Nonic",
        "function_str": "(x^9 + x^8 + ... + x)",
        "dim": 1,
        "suite": "extra",
        "source": "extra",
        "profile": "poly",
        "x_range": (-1.0, 1.0),
        "target": lambda x: x ** 9 + x ** 8 + x ** 7 + x ** 6 + x ** 5 + x ** 4 + x ** 3 + x ** 2 + x,
    },
    {
        "name": "Pagie-1",
        "function_str": "(1/(1+x^-4) + 1/(1+y^-4))",
        "dim": 2,
        "suite": "extra",
        "source": "extra",
        "profile": "multi",
        "x_range": (-5.0, 5.0),
        "y_range": (-5.0, 5.0),
        "target": lambda x, y: 1.0 / (1.0 + np.power(np.abs(x) + 1e-8, -4.0))
        + 1.0 / (1.0 + np.power(np.abs(y) + 1e-8, -4.0)),
    },
    {
        "name": "Trig-Mix-1",
        "function_str": "(sin(x) + 0.5*cos(2x))",
        "dim": 1,
        "suite": "extra",
        "source": "extra",
        "profile": "trig",
        "x_range": (-3.0, 3.0),
        "target": lambda x: np.sin(x) + 0.5 * np.cos(2 * x),
    },
    {
        "name": "Log-Sqrt-Mix",
        "function_str": "(log(x+2) + sqrt(x+2))",
        "dim": 1,
        "suite": "extra",
        "source": "extra",
        "profile": "log",
        "x_range": (0.0, 4.0),
        "target": lambda x: np.log(x + 2.0) + np.sqrt(x + 2.0),
    },
]

