import taichi as ti
ti.init(arch=ti.gpu)

import numpy as np
from Motifs import step, k, F, U, V, res, initialize
from PIL import Image, ImageOps



N_steps = 200
lr = 0.0001

def load_ref(path):
    img = Image.open(path).convert('L') 
    img = ImageOps.fit(img, (res, res))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.flip(arr, axis=1)    
    field = ti.field(dtype=ti.f32, shape=(res, res))
    field.from_numpy(arr)
    
    return field

girafe_ref = load_ref("Gray-Scott/img/girafe.jpg")

loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

@ti.kernel
def compute_loss():
    for i, j in V:
        diff = V[i, j] - girafe_ref[i, j]
        loss[None] += diff**2

F[None], k[None] = 0.03, 0.062
gui = ti.GUI("training...", res=(res * 2, res))

while gui.running:
    initialize()

    with ti.ad.Tape(loss=loss):
        for _ in range(N_steps):
            step()
        compute_loss()

    grad_F = F.grad[None]
    grad_k = k.grad[None]

    norm = np.sqrt(grad_F**2 + grad_k**2)

    F[None] -= lr * (grad_F / norm)
    k[None] -= lr * (grad_k / norm)

    F[None] = max(0.025, min(0.045, F[None]))
    k[None] = max(0.050, min(0.070, k[None]))
    
    display = np.concatenate([V.to_numpy(), girafe_ref.to_numpy()], axis=0)
    gui.set_image(display)
    gui.show()

    print(f"Loss: {loss[None]:.4f} | F: {F[None]:.4f} | k: {k[None]:.4f}")