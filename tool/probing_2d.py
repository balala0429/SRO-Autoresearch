import numpy as np


def get_probing_grid_2d(nx=32, ny=32, x_range=(-1.0, 1.0), y_range=(-1.0, 1.0)):
    """
    构造真正 2D 网格采样点并展开为长度 nx*ny 的向量。

    返回：
      - var_data: {'x': x_flat, 'y': y_flat}
      - x_flat, y_flat: shape=(nx*ny,)
    """
    x = np.linspace(x_range[0], x_range[1], nx, dtype=np.float32)
    y = np.linspace(y_range[0], y_range[1], ny, dtype=np.float32)
    xx, yy = np.meshgrid(x, y, indexing="xy")  # shape (ny, nx)
    x_flat = xx.reshape(-1)
    y_flat = yy.reshape(-1)
    return {"x": x_flat, "y": y_flat}, x_flat, y_flat

