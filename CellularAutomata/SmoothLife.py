
import taichi as ti
import taichi.math as tm

import time


#CONSTANTES
Ra = 12 #rayon interne
Rb = 3*Ra #rayon externe
b1, b2 = 0.257, 0.336 #zone de naissance
d1, d2 = 0.365, 0.549 #zone de survie
dt = 0.1
alpha_n = 0.028
alpha_m = 0.147

sim_w, sim_h = 512, 512

ti.init(arch=ti.gpu, random_seed=int(time.time()))

grid = ti.field(dtype=ti.f32, shape=(sim_w, sim_h))
grid_tmp = ti.field(dtype=ti.f32, shape=(sim_w, sim_h))

@ti.kernel
def init_random(grid: ti.template(), density: float):
    for i, j in grid:
        if ti.random() < density:
            grid[i, j] = ti.random()


@ti.kernel
def init_shapes(grid: ti.template(), cx: float, cy: float, r: float):

    for i, j in grid:
        if (i - cx)**2 + (j - cy)**2 < r**2: # Cercle de rayon 63
            grid[i, j] = ti.random()

@ti.func
def get_integrals(x, y, grid: ti.template()):
    sum_m = 0.0
    sum_n = 0.0
    count_m = 0.0
    count_n = 0.0
    
    limit = int(Rb) + 1
    
    Ra2 = Ra * Ra
    Rb2 = Rb * Rb

    for i in range(-limit, limit + 1):
        for j in range(-limit, limit + 1):
            
            d2 = float(i*i + j*j)
            
            if d2 >= Rb2: continue

            x_nb = (x + i + sim_w) % sim_w
            y_nb = (y + j + sim_h) % sim_h
            val = grid[x_nb, y_nb]
            
            if d2 < Ra2:
                sum_m += val
                count_m += 1.0
            else:
                sum_n += val
                count_n += 1.0
    m = 0.0
    if count_m > 0: m = sum_m / count_m
        
    n = 0.0
    if count_n > 0: n = sum_n / count_n
        
    return m, n

@ti.func
def sigma_1(x, a, alpha):
    return 1.0 / (1.0 + ti.exp(-4.0 * (x - a) / alpha))

@ti.func
def sigma_2(x, a, b):
    return sigma_1(x, a, alpha_n) * (1.0 - sigma_1(x, b, alpha_n))

@ti.func
def sigma_m(x, y, m):
    return x * (1 - sigma_1(m, 0.5, alpha_m)) + y * sigma_1(m, 0.5, alpha_m)

@ti.func
def s(n, m):
    return sigma_2(n, sigma_m(b1, d1, m), sigma_m(b2, d2, m))

@ti.kernel
def step(grid_r: ti.template(), grid_w: ti.template()):
    for i, j in grid_r:
        m, n = get_integrals(i, j, grid_r)

        target = s(n, m)

        grid_w[i, j] = grid_r[i, j] + (target - grid_r[i, j]) * dt
        

r = 80
for a in range(1, 2):
    cx = sim_w - a * 150
    cy = sim_h - a * 150
    init_shapes(grid, cx, cy, r)

init_random(grid, 0.045)

gui = ti.GUI("Smooth Life", res=(sim_w, sim_h))

while gui.running:
    step(grid, grid_tmp)
    grid, grid_tmp = grid_tmp, grid

    gui.set_image(grid)
    gui.show()

    


