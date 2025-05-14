import numpy as np

class AdvectionDiffusion2D:
    def __init__(self, nx, ny, dx, dy, u=5.0, v=0.0, diff=1e-5):
        self.nx, self.ny, self.dx, self.dy = nx, ny, dx, dy
        self.u, self.v, self.diff = u, v, diff
        self.T = np.zeros((nx, ny))

    def step(self, dt):
        T = self.T
        dTdx = (np.roll(T, -1, 0) - np.roll(T, 1, 0)) / (2 * self.dx)
        dTdy = (np.roll(T, -1, 1) - np.roll(T, 1, 1)) / (2 * self.dy)
        lap = (
            (np.roll(T, -1, 0) + np.roll(T, 1, 0) - 2 * T) / self.dx**2 +
            (np.roll(T, -1, 1) + np.roll(T, 1, 1) - 2 * T) / self.dy**2
        )
        self.T += (-self.u * dTdx - self.v * dTdy + self.diff * lap) * dt
