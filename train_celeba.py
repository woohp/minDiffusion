import os
from typing import Optional

import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from mindiffusion.ddpm import DDPM
from mindiffusion.unet import NaiveUnet

load_dotenv("./.env")
CELEBA_PATH = os.getenv("CELEBA_PATH")


def train_celeba(n_epoch: int = 100, device: str = "cuda:1", load_pth: Optional[str] = None) -> None:
    ddpm = DDPM(eps_model=NaiveUnet(3, 3, n_feat=128), betas=(1e-4, 0.02), n_T=1000)

    if load_pth is not None:
        ddpm.load_state_dict(torch.load("ddpm_celeba.pth"))

    ddpm.to(device)

    tf = transforms.Compose([  # resize to 512 x 512, convert to tensor, normalize
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = ImageFolder(
        root=CELEBA_PATH,
        transform=tf,
    )

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=20)
    optim = torch.optim.Adam(ddpm.parameters(), lr=2e-5)

    for i in range(n_epoch):
        print(f"Epoch {i} : ")
        ddpm.train()

        pbar = tqdm(dataloader)
        loss_ema = None
        for x, _ in pbar:
            optim.zero_grad()
            x = x.to(device)
            loss = ddpm(x)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()

        ddpm.eval()
        with torch.no_grad():
            xh = ddpm.sample(8, (3, 128, 128), device)
            xset = torch.cat([xh, x[:8]], dim=0)
            grid = make_grid(xset, normalize=True, value_range=(-1, 1), nrow=4)
            save_image(grid, f"./contents/ddpm_sample_celeba{i:03d}.png")

            # save model
            torch.save(ddpm.state_dict(), "./ddpm_celeba.pth")


if __name__ == "__main__":
    train_celeba()
