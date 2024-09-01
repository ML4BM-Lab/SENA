import numpy as np
import torch
import pygame
from pygame import gfxdraw, init
from typing import Callable, Optional
import cv2
import os

COLOURS_ = [
    [2, 156, 154],
    [222, 100, 100],
    [149, 59, 123],
    [74, 114, 179],
    [27, 159, 119],
    [218, 95, 2],
    [117, 112, 180],
    [232, 41, 139],
    [102, 167, 30],
    [231, 172, 2],
    [167, 118, 29],
    [102, 102, 102],
]

SCREEN_DIM = 64


def circle(
    x_,
    y_,
    surf,
    color=(204, 204, 0),
    radius=0.1,
    screen_width=SCREEN_DIM,
    y_shift=0.0,
    offset=None,
):
    if offset is None:
        offset = screen_width / 2
    scale = screen_width
    x = scale * x_ + offset
    y = scale * y_ + offset

    gfxdraw.aacircle(
        surf, int(x), int(y - offset * y_shift), int(radius * scale), color
    )
    gfxdraw.filled_circle(
        surf, int(x), int(y - offset * y_shift), int(radius * scale), color
    )


class Balls(torch.utils.data.Dataset):
    ball_rad = 0.04
    screen_dim = 64

    def __init__(
        self,
        transform: Optional[Callable] = None,
        n_balls: int = 1,
    ):
        super(Balls, self).__init__()
        if transform is None:

            def transform(x):
                return x

        self.transform = transform
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_dim, self.screen_dim))
        self.surf = pygame.Surface((self.screen_dim, self.screen_dim))
        self.n_balls = n_balls

    def __len__(self) -> int:
        # arbitrary since examples are generated online.
        return 20000

    def draw_scene(self, z):
        self.surf.fill((255, 255, 255))
        if z.ndim == 1:
            z = z.reshape((1, 2))
        for i in range(z.shape[0]):
            circle(
                z[i, 0],
                z[i, 1],
                self.surf,
                color=COLOURS_[i],
                radius=self.ball_rad,
                screen_width=self.screen_dim,
                y_shift=0.0,
                offset=0.0,
            )
        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
        )

    def __getitem__(self, item):
        raise NotImplemented()

class BlockOffset(Balls):
    def __init__(
        self,
        transform: Optional[Callable] = None,
        n_balls: int = 1,
        true_mech: bool = False,
        combination_offsets: bool = False,
    ):
        super().__init__(transform=transform, n_balls=n_balls)
        guess = np.random.uniform(0.04, 0.1, size=(n_balls, 2))
        self.guess = guess
        self.true_mech = true_mech
        self.combination_offsets = combination_offsets

    def __getitem__(self, item):
        z1 = np.random.uniform(0.1, 0.9, size=(self.n_balls, 2))
        x1 = self.draw_scene(z1)
        x1 = self.transform(x1)

        loc = np.random.randint(self.n_balls * 2)
        if self.combination_offsets:
            offset_loc = np.random.choice(self.n_balls * 2, 2)
        else:
            offset_loc = [loc % (self.n_balls * 2), (loc + 1) % (self.n_balls * 2)]
        b = np.zeros(self.n_balls * 2)
        b[offset_loc] = 0.05
        b = b.reshape((self.n_balls, 2))
        z2 = z1 + b
        x2 = self.draw_scene(z2)
        x2 = self.transform(x2)

        if self.true_mech:
            b_guess = b
        else:
            b_guess = np.zeros(self.n_balls * 2)
            b_guess[offset_loc] = self.guess.flatten()[offset_loc]
            b_guess = b_guess.reshape(z2.shape)
        b = torch.tensor(b_guess).float().flatten().squeeze()
        A = torch.eye(2)
        return (z1.flatten(), z2.flatten()), (x1, x2), (A, b)

class SparseBall(Balls):
    def __init__(
        self,
        transform: Optional[Callable] = None,
        n_balls: int = 1,
        n_offsets: int = 2,
        true_mech: bool = False,
    ):
        super().__init__(transform=transform, n_balls=n_balls)
        self.n_offsets = n_offsets
        self.offsets = np.array([[0.0, 1.0], [1.0, 0.0]]) * 0.05
        guess = np.random.uniform(0.04, 0.1, size=(n_balls, n_offsets, 2))
        self.guess = guess
        self.true_mech = true_mech

    def __getitem__(self, item):
        z1 = np.random.uniform(0.1, 0.9, size=(self.n_balls, 2))
        x1 = self.draw_scene(z1)
        x1 = self.transform(x1)
        ball_id = np.random.randint(self.n_balls)
        coo_id = np.random.randint(2)
        offset_id = np.random.randint(self.n_offsets)
        b = np.zeros_like(z1)
        b[ball_id, coo_id] = 0.05 if offset_id == 1 else -0.05
        z2 = z1 + b
        x2 = self.draw_scene(z2)
        x2 = self.transform(x2)
        if self.true_mech:
            b_guess = b
        else:
            b_guess = np.zeros_like(z1)
            b_guess[ball_id, coo_id] = self.guess[ball_id, offset_id, coo_id]
        b = torch.tensor(b_guess).float().flatten().squeeze()
        A = torch.eye(2)
        return (z1.flatten(), z2.flatten()), (x1, x2), (A, b)
    

## get sparseballs
dataset = SparseBall(n_balls = 4)

for ball in dataset:
    (z1, z2), (x1, x2), (A, b) = ball

    #save x1 image
    cv2.imwrite(os.path.join('crl','env1_exp','figs','img1.png'), x1)
    cv2.imwrite(os.path.join('crl','env1_exp','figs','img2.png'), x2)
    break