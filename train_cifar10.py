#!/usr/bin/env python
import click
import torch
from torch import Tensor
from torchext import engine
from torchext.callbacks import Callback, ModelCheckpoint
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid, save_image

from mindiffusion.ddpm import DDPM
from mindiffusion.unet import NaiveUnet


def preprocess_data(x: Tensor, y: Tensor) -> tuple[tuple[Tensor, Tensor], Tensor]:
    eps = torch.randn_like(x)  # eps ~ N(0, 1)
    return (x, eps), eps


@click.command()
@click.option("--n-epoch", type=int, default=100)
@click.option("--load-pth", type=click.Path())
def train_cifar10(n_epoch: int = 100, load_pth: str | None = None) -> None:
    ddpm = DDPM(eps_model=NaiveUnet(3, 3, n_feat=128), betas=(1e-4, 0.02), n_T=1000)

    if load_pth is not None:
        ddpm.load_state_dict(torch.load("ddpm_cifar.pth"))

    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = CIFAR10("./data", train=True, download=True, transform=tf)
    optim = torch.optim.Adam(ddpm.parameters(), lr=1e-5)

    class SaveSample(Callback):
        def on_epoch_end(self, epoch: int, log: dict[str, float | list[float]]) -> None:
            ddpm: DDPM = self.model
            ddpm.eval()
            with torch.no_grad():
                xh = ddpm.sample(8, (3, 32, 32))
            xset = xh
            # xset = torch.cat([xh, x[:8]], dim=0)
            grid = make_grid(xset, normalize=True, value_range=(-1, 1), nrow=4)
            save_image(grid, f"./contents/ddpm_sample_cifar{epoch}.png")

            # save model
            torch.save(ddpm.state_dict(), "./ddpm_cifar.pth")

    engine.train(
        ddpm,
        train_data=dataset,
        criterion=torch.nn.MSELoss(),
        batch_size=512,
        optimizer=optim,
        epochs=n_epoch,
        preprocess_data=preprocess_data,
        callbacks=[SaveSample(), ModelCheckpoint("./ddpm_cifar.pth")],
    )


if __name__ == "__main__":
    train_cifar10()
