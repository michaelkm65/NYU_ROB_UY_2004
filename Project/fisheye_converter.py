"""
Minimal fisheye to equirectangular converter for Hailo detection
"""
import cv2
import numpy as np
import yaml


class DoubleSphereModel:
    """Double sphere fisheye camera model"""
    def __init__(self, cx, cy, fx, fy, xi, alpha, width=None, height=None):
        self.cx = cx
        self.cy = cy
        self.fx = fx
        self.fy = fy
        self.xi = xi
        self.alpha = alpha
        self.width = width
        self.height = height

    def project(self, x, y, z, eps=1e-9):
        r2 = x * x + y * y
        d1 = np.sqrt(r2 + z * z)
        k2 = self.xi * d1 + z
        d2 = np.sqrt(r2 + k2 * k2)
        denom_raw = self.alpha * d2 + (1.0 - self.alpha) * k2

        valid = denom_raw > 0
        denom = np.maximum(denom_raw, eps)

        mx = x / denom
        my = y / denom

        u = self.fx * mx + self.cx
        v = self.fy * my + self.cy

        return u, v, valid


def create_equirectangular_rays(width, height, h_fov_deg=220.0):
    """Generate 3D rays for equirectangular projection"""
    h_fov_rad = np.deg2rad(h_fov_deg)

    lon = (np.linspace(0, width - 1, width) / (width - 1)) * h_fov_rad - (h_fov_rad / 2)
    lat = (np.linspace(0, height - 1, height) / (height - 1)) * np.pi - (np.pi / 2)
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    x = np.cos(lat_grid) * np.sin(lon_grid)
    y = np.sin(lat_grid)
    z = np.cos(lat_grid) * np.cos(lon_grid)

    return x, y, z


def fisheye_to_equirectangular(img, model, out_width, out_height=None, h_fov_deg=220.0):
    """Convert fisheye image to equirectangular projection"""
    if out_height is None:
        out_height = out_width // 2

    x, y, z = create_equirectangular_rays(out_width, out_height, h_fov_deg)
    u, v, valid = model.project(x, y, z)

    map_x = u.astype(np.float32)
    map_y = v.astype(np.float32)

    equirect = cv2.remap(
        img, map_x, map_y, 
        interpolation=cv2.INTER_LINEAR, 
        borderMode=cv2.BORDER_CONSTANT, 
        borderValue=(0, 0, 0)
    )

    if equirect.ndim == 3:
        equirect[~valid] = (0, 0, 0)
    else:
        equirect[~valid] = 0

    return equirect


def load_camera_model(config_path, img_width, img_height):
    """Load camera parameters and create model"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    params = config.get("camera_params", {})
    return DoubleSphereModel(
        cx=params["cx"],
        cy=params["cy"],
        fx=params["fx"],
        fy=params["fy"],
        xi=params["xi"],
        alpha=params["alpha"],
        width=img_width,
        height=img_height,
    )



