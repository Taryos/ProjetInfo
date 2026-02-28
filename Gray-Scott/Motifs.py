import taichi as ti

if not ti.lang.impl.get_runtime().prog:
    ti.init(arch=ti.gpu)

res = 128
Du, Dv = 0.2, 0.1
dt = 0.1

U = ti.field(dtype=ti.f32, shape=(res, res), needs_grad=True)
V = ti.field(dtype=ti.f32, shape=(res, res), needs_grad=True)
U_new = ti.field(dtype=ti.f32, shape=(res, res), needs_grad=True)
V_new = ti.field(dtype=ti.f32, shape=(res, res), needs_grad=True)

F = ti.field(ti.f32, shape=(), needs_grad=True)
k = ti.field(ti.f32, shape=(), needs_grad=True)

@ti.kernel
def initialize():
    for i, j in V:
        U[i, j] = 1.0  
        V[i, j] = 0.0

        if ti.random() > 0.95:
            V[i, j] = 0.5

@ti.func
def laplacian(C, i, j):
    return (
        C[i-1, j] + C[i+1, j] + C[i, j-1] + C[i, j+1] - 4 * C[i, j]
    )

@ti.kernel
def step():
    for i, j in U:
        if 1 <= i < res-1 and 1 <= j < res-1:
            reaction = U[i, j] * V[i, j] * V[i, j]
            U_new[i, j] = U[i, j] + dt * (Du * laplacian(U, i, j) - reaction + F[None] * (1 - U[i, j]))
            V_new[i, j] = V[i, j] + dt * (Dv * laplacian(V, i, j) + reaction - (F[None] + k[None]) * V[i, j])

    for i, j in U:
        U[i, j] = max(0.0, min(1.0, U_new[i, j]))
        V[i, j] = max(0.0, min(1.0, V_new[i, j]))

if __name__ == "__main__":
    F[None], k[None] = 0.035, 0.060
    initialize()
    gui = ti.GUI("Gray-Scott : Preview", res=(res, res))
    while gui.running:
        for _ in range(10): step()
        gui.set_image(V)
        gui.show()