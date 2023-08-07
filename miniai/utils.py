import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter


def ensure_reproducity():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)


def _show_data(X, Y):
    circle = np.where(Y[:, 0] == 1)
    mark = np.where(Y[:, 1] == 1)
    plt.clf()
    plt.figure()  # Create figure
    ax = plt.gca()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    ax.scatter(X[mark, 0], X[mark, 1], marker='x', c='blue')
    ax.scatter(X[circle, 0], X[circle, 1], marker='o', c='red')
    return ax


def show_data(X, Y):
    _show_data(X, Y)
    plt.savefig('./img/higham-data.png', bbox_inches='tight', pad_inches=0)
    plt.show()


def show_costs(savecost):
    plt.clf()
    plt.figure()  # Create figure
    plt.yscale('log', basey=10)
    plt.plot(savecost)
    plt.savefig('./img/higham-savecost.png', bbox_inches='tight', pad_inches=0)
    plt.show()


def show_decision_boundary(X, Y, h):
    ax = _show_data(X, Y)
    X1, X2 = np.meshgrid(np.linspace(0, 1), np.linspace(0, 1))
    Mesh_X = np.column_stack((X1.flatten(), X2.flatten()))
    Mesh_Y = h(Mesh_X).T
    Mesh_Y = Mesh_Y[:, 0] / np.sum(Mesh_Y, axis=1)
    Mesh_Y = Mesh_Y.reshape(X1.shape)
    ax.contour(X1, X2, Mesh_Y, levels=[0.5], linewidths=1, colors='black')
    plt.savefig('./img/higham-decision-boundary.png',
                bbox_inches='tight', pad_inches=0)
    plt.show()


class Visualizer(object):
    def __init__(self):
        self.epochs = []
        self.train_losses = []
        self.val_losses = []

    def plot(self):
        plt.ion()  # Enable interactive mode
        plt.clf()
        self.fig = plt.figure()  # Create figure
        # Add subplot (dont worry only one plot appears)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_autoscale_on(True)  # enable autoscale
        self.ax.set_yscale('log')
        self.train_cur, = plt.plot(
            self.epochs, self.train_losses, 'o-', label='train', markevery=[-1])
        if self.val_losses:
            self.val_cur, = plt.plot(
                self.epochs, self.val_losses, 'o-', label='validation', markevery=[-1])
        self.ax.relim()  # Recalculate limits
        self.ax.autoscale_view(True, True, True)  # Autoscale
        # draw and show it
        plt.show()

    def update(self, t, train_loss, val_loss=None):
        self.epochs.append(t)
        self.train_losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)


def collect_losses(it, f, parameters):
    losses = []
    for epoch in it:
        losses.append(f(parameters).item())
    return losses


def collect_losses_and_log_to_tensorboard(it, cost_function, parameters):
    writer = SummaryWriter()
    losses = []
    for epoch in it:
        loss = cost_function(parameters)
        losses.append(loss)
        writer.add_scalar("Loss/train", loss, epoch)
    writer.flush()
    writer.close()
    return losses
