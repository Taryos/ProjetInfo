import taichi as ti
import taichi.math as tm
import numpy as np
import time

ti.init(arch=ti.gpu)

ZOOM = 5

sim_w, sim_h = 200, 150

disp_w, disp_h = sim_w * ZOOM, sim_h * ZOOM


grid = ti.field(dtype=ti.u8, shape=(sim_w, sim_h))
grid_tmp = ti.field(dtype=ti.u8, shape=(sim_w, sim_h))

display = ti.field(dtype=ti.u8, shape=(disp_w, disp_h))

@ti.func
def count_neighbors(x, y, grid: ti.template()):
    s = -grid[x, y]
    for i in range(-1, 2):
        for j in range(-1, 2):
            x_nb = (x + i) % sim_w
            y_nb = (y + j) % sim_h

            s += grid[x_nb, y_nb]
    return s

@ti.kernel
def init_random(grid: ti.template(), density: float):
    for i, j in grid:
        if ti.random() < density: 
            grid[i, j] = 1
        else:
            grid[i, j] = 0

@ti.kernel
def step(grid_r: ti.template(), grid_w: ti.template()):
    for i, j in grid_r:
        nb = count_neighbors(i, j, grid_r)
        if grid_r[i, j] == 1:
            if nb < 2 or nb > 3:
                grid_w[i, j] = 0
            else:
                grid_w[i, j] = 1
        else:
            if nb == 3:
                grid_w[i, j] = 1
            else:
                grid_w[i, j] = 0
@ti.kernel
def render_zoom(src: ti.template(), dest: ti.template()):
    for x, y in dest:
        src_x = x // ZOOM
        src_y = y // ZOOM
        dest[x, y] = src[src_x, src_y]
        val = src[src_x, src_y]
        dest[x, y] = val * 255

init_random(grid, 0.2)

gui = ti.GUI("Game of Life", res=(disp_w, disp_h))

while gui.running:
    step(grid, grid_tmp)

    render_zoom(grid, display)

    arr = display.to_numpy()
    gui.set_image(display)
    gui.show()
    time.sleep(0.05)

    grid, grid_tmp = grid_tmp, grid

init_random(grid, 0.2)


