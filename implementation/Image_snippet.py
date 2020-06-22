import torch as th
import pro_gan_pytorch.PRO_GAN as pg
import matplotlib.pyplot as plt

device = th.device("cuda" if th.cuda.is_available()
                   else "cpu")
gen = pg.Generator(depth=4, latent_size=128,
                   use_eql=False).to(device)
gen.load_state_dict(th.load("training_runs/haiti/saved_models/GAN_GEN_3.pth"))
noise = th.randn(1,128).to(device)
sample_image = gen(noise, detph=3, alpha=1).detach()
plt.imshow(sample_image[0].permute(1,2,0)/2+0.5)
plt.show()