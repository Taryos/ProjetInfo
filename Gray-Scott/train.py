import taichi as ti
ti.init(arch=ti.gpu)

import numpy as np
from PIL import Image, ImageOps, ImageFilter

from Motifs import step, k, F, U, V, res, initialize

N_steps = 200
N_prior = 100

lr = 0.00001

loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
display = ti.field(dtype=ti.f32, shape=(res * 2, res))

def load_ref(path):
    img = Image.open(path).convert('L')
    img = ImageOps.fit(img, (res, res))
    
    img = img.filter(ImageFilter.MedianFilter(size=15))

    arr = np.array(img).astype(np.float32) / 255.0
    
    moyenne = np.mean(arr)
    seuil = moyenne * 1.15 
    arr = np.where(arr > seuil, 1.0, 0.0)
    
    arr = np.flip(arr, axis=1) 
    
    field = ti.field(dtype=ti.f32, shape=(res, res))
    field.from_numpy(arr)
    return field

cible_ref = load_ref("Gray-Scott/img/girafe.jpg") 

@ti.kernel
def compute_loss():
    for i, j in V:
        diff = cible_ref[i, j] - V[i, j]
        loss[None] += (diff**2) / (res * res)

@ti.kernel
def update_display():
    for i, j in V:
        display[i, j] = V[i, j]
        display[i + res, j] = cible_ref[i, j]

# --- DÉBUT DE L'ENTRAÎNEMENT ---
F[None], k[None] = 0.055, 0.062
initialize()

window = ti.ui.Window("Training Gray-Scott (Global Params)", (res * 2, res))
canvas = window.get_canvas()

epoch = 0

while window.running:
    for _ in range(N_prior):
        step()
        
    with ti.ad.Tape(loss=loss):
        for _ in range(N_steps):
            step()
        compute_loss()

    F_new = F[None] - lr * F.grad[None]
    k_new = k[None] - lr * k.grad[None]
    
    F[None] = max(0.0, min(1.0, F_new))
    k[None] = max(0.0, min(1.0, k_new))

    epoch += 1
    print(f"Epoch {epoch} | Loss: {loss[None]:.6f} | F: {F[None]:.5f} | k: {k[None]:.5f}")
    
    update_display()
    canvas.set_image(display)
    window.show()