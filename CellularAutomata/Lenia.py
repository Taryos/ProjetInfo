import taichi as ti
import time
import numpy as np
import cupy as cp

#CONSTANTES
sim_w, sim_h = 512, 512
R = 13.0
dt = 0.02

ti.init(arch=ti.gpu, random_seed=int(time.time()))

grid = ti.field(ti.f32, shape=(sim_w, sim_h))
potential = ti.field(ti.f32, shape=(sim_w, sim_h))
kernel = ti.field(ti.f32, shape=(sim_w, sim_h))

@ti.kernel
def build_kernel():
    mu = 0.5
    sigma = 0.15
    for i, j in kernel:
        dx = float(i)
        dy = float(j)

        if dx > sim_w / 2.0: dx -= sim_w
        if dy > sim_h / 2.0: dy -= sim_h

        dist = ti.sqrt(dx**2 + dy**2) / R

        if dist <= 1.0:
            kernel[i, j] = ti.exp(-(dist - mu)**2 / (2 * sigma**2))
        else:
            kernel[i, j] = 0.0

build_kernel()
K_cp = cp.asarray(kernel.to_numpy())
K_cp /= K_cp.sum()
kernel_fft = cp.fft.fft2(K_cp)

@ti.kernel
def init_random(density: float):
    for i, j in grid:
        if ti.random() < density:
            grid[i, j] = ti.random()

@ti.kernel
def init_shapes(cx: float, cy: float, r: float):
    for i, j in grid:
        if (i - cx)**2 + (j - cy)**2 < r**2: # Cercle de rayon r
            grid[i, j] = ti.random()

@ti.func
def growth(u):
    mu = 0.15
    sigma = 0.015
    return -1.0 + 2.0 * ti.exp(-(u - mu)**2 / (2 * sigma * sigma))

@ti.kernel
def step():
    for i, j in grid:
        u = potential[i, j]
        v = grid[i, j] + dt * growth(u)
        grid[i, j] = ti.max(0.0, ti.min(1.0, v))

X_cp = cp.zeros((sim_h, sim_w), dtype=cp.float32)

def fft_convolve():
    X_cp[:] = cp.asarray(grid.to_numpy())
    grid_fft = cp.fft.fft2(X_cp) #fft de la grille
    result_fft = grid_fft * kernel_fft #convolution avec le noyau
    U_cp = cp.real(cp.fft.ifft2(result_fft)) #fft inverse
    potential.from_numpy(cp.asnumpy(U_cp))

init_random(0.205457)

window = ti.ui.Window("Lenia GGUI", (sim_w, sim_h))
canvas = window.get_canvas()

while window.running:
    fft_convolve()
    step()
    canvas.set_image(grid)
    window.show()

