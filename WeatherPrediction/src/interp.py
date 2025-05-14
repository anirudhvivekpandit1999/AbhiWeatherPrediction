import numpy as np
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler

def interpolate_stations_to_grid(stations, temps, grid_lat, grid_lon, k=4):
    M = stations.shape[0]
    if M == 0:
        raise ValueError("No station locations provided for interpolation.")
    if M == 1:
        return np.full(grid_lat.shape, temps[0])

    k_eff = min(k, M)
    pts = np.vstack([grid_lat.ravel(), grid_lon.ravel()]).T
    tree = cKDTree(stations)
    dists, idx = tree.query(pts, k=k_eff)
    if k_eff == 1:
        dists = dists[:, None]
        idx = idx[:, None]
    w = 1.0 / (dists + 1e-6)
    w /= w.sum(axis=1, keepdims=True)
    vals = np.sum(temps[idx] * w, axis=1)
    return vals.reshape(grid_lat.shape)

def scale_features(df):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)
    return scaled, scaler
