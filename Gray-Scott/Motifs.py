import taichi as ti

if getattr(ti.lang.impl.get_runtime(), 'prog', None) is None:
    ti.init(arch=ti.gpu)

res = 512
Du, Dv = 0.1, 0.05
dt = 1.0

U = None; V = None; U_new = None; V_new = None; F = None; k = None

def setup_fields(training=False):
    global U, V, U_new, V_new, F, k
    print(f"[Gray-Scott] Allocation de la mémoire... (Mode Entraînement: {training})")
    
    U = ti.field(dtype=ti.f32, shape=(res, res), needs_grad=training)
    V = ti.field(dtype=ti.f32, shape=(res, res), needs_grad=training)
    U_new = ti.field(dtype=ti.f32, shape=(res, res), needs_grad=training)
    V_new = ti.field(dtype=ti.f32, shape=(res, res), needs_grad=training)
    
    # RETOUR À LA NORMALE : shape=() pour un paramètre global
    F = ti.field(ti.f32, shape=(), needs_grad=training)
    k = ti.field(ti.f32, shape=(), needs_grad=training)

is_training = (__name__ != "__main__")
setup_fields(training=is_training)

@ti.kernel
def initialize_random():
    for i, j in V:
        U[i, j] = 1.0  
        V[i, j] = 0.0
        if ti.random() > 0.9:
            V[i, j] = 0.5

@ti.kernel
def initialize():
    for i, j in V:
        U[i, j] = 1.0  
        V[i, j] = 0.0
        if res//2 - 5 < i < res//2 + 5 and res//2 - 5 < j < res//2 + 5:
            V[i, j] = 1.0
            U[i, j] = 0.0

@ti.func
def lapmod(C: ti.template(), i, j):
    return (C[(i-1+res)%res, j] + C[(i+1)%res, j] + 
            C[i, (j-1+res)%res] + C[i, (j+1)%res] - 4.0 * C[i, j])

@ti.kernel
def step():
    for i, j in U:
        reaction = U[i, j] * V[i, j] * V[i, j]
        U_new[i, j] = U[i, j] + dt * (Du * lapmod(U, i, j) - reaction + F[None] * (1.0 - U[i, j]))
        V_new[i, j] = V[i, j] + dt * (Dv * lapmod(V, i, j) + reaction - (F[None] + k[None]) * V[i, j])

    for i, j in U:
        U[i, j] = ti.max(0.0, ti.min(1.0, U_new[i, j]))
        V[i, j] = ti.max(0.0, ti.min(1.0, V_new[i, j]))

if __name__ == "__main__":
    F[None], k[None] = 0.055, 0.062
    initialize()
    
    window = ti.ui.Window("Gray-Scott", (res, res))
    canvas = window.get_canvas()
    
    while window.running:
        for _ in range(10): 
            step()
        canvas.set_image(V)
        window.show()