
import taichi as ti
import time
import numpy as np
import cupy as cp



#CONSTANTES
sim_w, sim_h = 512, 512
R = 13 
K = 2 * R + 1 
dt = 0.05

mu = 0.5
sigma = 0.15


y, x = np.ogrid[-R:R+1, -R:R+1] #creation de 2 vecteurs pour faire le tableau du kernel
r = np.sqrt(x*x + y*y) / R #broadcasting puis normalisée avec / R

K_np = np.exp(-(r - mu)**2 / (2 * sigma**2)) #poids du kernel
K_np[r > 1] = 0.0 # masking : on veut un disque donc comme r normalisé si r>1 on est hors du disque
K_np /= K_np.sum() #normalisation

kernel = cp.zeros((sim_h, sim_w), dtype=cp.float32) #tableau rempli de 0 de taille sim_h, sim_w nécessaire pour fft
kernel[:K, :K] = cp.asarray(K_np) #on met le petit kernel de rayon R dans le grand tableau en haut à gauche pour fft

kernel = cp.roll(kernel, -R, axis=0)
kernel = cp.roll(kernel, -R, axis=1) #centrage du centre du kernel pour appliquer fft

kernel /= kernel.sum() #normalisation 
kernel_fft = cp.fft.fft2(kernel) #on applique la fft

ti.init(arch=ti.gpu, random_seed=int(time.time()))

grid = ti.field(ti.f32, shape=(sim_w, sim_h, 3))
potential = ti.field(ti.f32, shape=(sim_w, sim_h, 3))

@ti.kernel
def init_random(density: float):
    for i, j, c in grid:
        if ti.random() < density:
            grid[i, j, 0] = ti.random()
            grid[i, j, 1] = ti.random()
            grid[i, j, 2] = ti.random()

@ti.kernel
def init_shapes(cx: float, cy: float, r: float):
    for i, j, c in grid:
        if (i - cx)**2 + (j - cy)**2 < r**2: # Cercle de rayon r
            grid[i, j] = ti.random()

@ti.func
def growth(u):
    mu = 0.15
    sigma = 0.015
    return -1.0 + 2.0 * ti.exp(-(u - mu)**2 / (2 * sigma * sigma))

@ti.kernel
def step():
    for i, j, c in grid:
        u = potential[i, j, c]
        v = grid[i, j, c] + dt * growth(u)
        grid[i, j, c] = ti.max(0.0, ti.min(1.0, v)) #on s'assure de rester en 0 et 1

X_cp = cp.zeros((sim_h, sim_w, 3), dtype=cp.float32)

def fft_convolve():
    X_cp[:] = cp.asarray(grid.to_numpy())
    grid_fft = cp.fft.fft2(X_cp, axes=(0, 1)) #fft de la grille
    result_fft = grid_fft * kernel_fft[:, :, cp.newaxis] #convolution avec le noyau
    U_cp = cp.real(cp.fft.ifft2(result_fft, axes=(0, 1))) #fft inverse
    potential.from_numpy(cp.asnumpy(U_cp))


init_random(0.075)

gui = ti.GUI("Lenia", res=(sim_w, sim_h))

while gui.running:
    fft_convolve()
    step()
    gui.set_image(grid)
    gui.show()

