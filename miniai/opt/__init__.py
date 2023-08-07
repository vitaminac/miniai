# The code is adapted from
# https://pytorch.org/docs/stable/optim.html
# https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
from .gd import GradientDescent
from .momentum import GradientDescentWithMomentum
from .adagrad import AdaGrad
from .rmspro import RMSProp
from .adam import Adam
from .function import Function, DifferentiableFunction
