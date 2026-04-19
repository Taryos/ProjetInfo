
import taichi as ti
import time


#CONSTANTES
Ra = 12.0 #rayon interne
Rb = 3.0*Ra #rayon externe
b1, b2 = 0.257, 0.336 #zone de naissance
d1, d2 = 0.365, 0.549 #zone de survie
dt = 0.1
alpha_n = 0.028
alpha_m = 0.147

sim_w, sim_h = 512, 512

ti.init(arch=ti.gpu, random_seed=int(time.time()))

grid = ti.field(dtype=ti.f32, shape=(sim_w, sim_h, 2))
display = ti.field(dtype=ti.f32, shape=(sim_w, sim_h))

@ti.kernel
def init_random(density: float):
    for i, j in ti.ndrange(sim_w, sim_h):
        if ti.random() < density:
            grid[i, j, 0] = ti.random()
        else:
            grid[i, j, 0] = 0.0


@ti.kernel
def init_shapes(cx: float, cy: float, r: float):
    for i, j in ti.ndrange(sim_w, sim_h):
        if (i - cx)**2 + (j - cy)**2 < r**2:
            grid[i, j, 0] = ti.random()

@ti.func
def get_integrals(x, y, phase: ti.i32):
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

            x_nb = x + i
            y_nb = y + j

            if x_nb < 0: x_nb += sim_w
            elif x_nb >= sim_w: x_nb -= sim_w
            
            if y_nb < 0: y_nb += sim_h
            elif y_nb >= sim_h: y_nb -= sim_h

            val = grid[x_nb, y_nb, phase]
            
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
def step(phase: ti.i32):
    next_phase = 1 - phase
    for i, j in ti.ndrange(sim_w, sim_h):
        m, n = get_integrals(i, j, phase)

        target = s(n, m)

        grid[i, j, next_phase] = grid[i, j, phase] + (target - grid[i, j, phase]) * dt
        
@ti.kernel
def update_display(phase: ti.i32):
    for i, j in display:
        display[i, j] = grid[i, j, phase]

r = 87
for a in range(1, 2):
    cx = sim_w - a * 150
    cy = sim_h - a * 150
    init_shapes(cx, cy, r)



window = ti.ui.Window("Smooth Life GGUI", (sim_w, sim_h))
canvas = window.get_canvas()

phase = 0

while window.running:
    step(phase)
    
    update_display(1 - phase)

    canvas.set_image(display)
    window.show()

    phase = 1 - phase
    


