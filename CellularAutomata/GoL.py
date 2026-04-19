import taichi as ti

ti.init(arch=ti.gpu)

ZOOM = 5
sim_w, sim_h = 200, 150
disp_w, disp_h = sim_w * ZOOM, sim_h * ZOOM

grid = ti.field(dtype=ti.i32, shape=(sim_w, sim_h, 2))

display = ti.field(dtype=ti.f32, shape=(disp_w, disp_h))

@ti.func
def count_neighbors(x, y, phase: ti.i32):
    s = 0
    for i in ti.static(range(-1, 2)):
        for j in ti.static(range(-1, 2)):
            if i != 0 or j != 0:
                nx = x + i
                ny = y + j
                
                if nx < 0: nx += sim_w
                elif nx >= sim_w: nx -= sim_w
                
                if ny < 0: ny += sim_h
                elif ny >= sim_h: ny -= sim_h
                
                s += grid[nx, ny, phase]
    return s

@ti.kernel
def init_random(density: float):
    for i, j in ti.ndrange(sim_w, sim_h):
        if ti.random() < density: 
            grid[i, j, 0] = 1
        else:
            grid[i, j, 0] = 0

@ti.kernel
def step(phase: ti.i32):
    next_phase = 1 - phase
    for i, j in ti.ndrange(sim_w, sim_h):
        nb = count_neighbors(i, j, phase)
        state = grid[i, j, phase]
        
        if state == 1 and (nb == 2 or nb == 3):
            grid[i, j, next_phase] = 1
        elif state == 0 and nb == 3:
            grid[i, j, next_phase] = 1
        else:
            grid[i, j, next_phase] = 0

@ti.kernel
def update_display(phase: ti.i32):
    for i, j in display:
        src_i = i // ZOOM
        src_j = j // ZOOM
        display[i, j] = float(grid[src_i, src_j, phase])

init_random(0.2)

window = ti.ui.Window("Game of Life - Vulkan/CUDA", (disp_w, disp_h), fps_limit=30)
canvas = window.get_canvas()

phase = 0

while window.running:
    step(phase)
    
    update_display(1 - phase)
    
    # Le canvas gère le ZOOM matériellemment et instantanément
    canvas.set_image(display)
    window.show()
    
    phase = 1 - phase
    
