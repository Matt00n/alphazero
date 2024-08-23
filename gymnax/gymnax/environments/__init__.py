from .environment import EnvParams, EnvState
from .classic_control import (
    # Pendulum,
    CartPole,
    MountainCar,
    # ContinuousMountainCar,
    Acrobot,
)

from .minatar import (
    MinAsterix,
    MinBreakout,
    MinFreeway,
    MinSeaquest,
    MinSpaceInvaders,
)

from .custom import (
    ProcMaze
)

# from .misc import (
#     BernoulliBandit,
#     GaussianBandit,
#     FourRooms,
#     MetaMaze,
#     PointRobot,
#     Reacher,
#     Swimmer,
#     Pong,
# )


__all__ = [
    "EnvParams",
    "EnvState",
    "Pendulum",
    "CartPole",
    "MountainCar",
    "ContinuousMountainCar",
    "Acrobot",
    "Catch",
    "DeepSea",
    "DiscountingChain",
    "MemoryChain",
    "UmbrellaChain",
    "MNISTBandit",
    "SimpleBandit",
    "MinAsterix",
    "MinBreakout",
    "MinFreeway",
    "MinSeaquest",
    "MinSpaceInvaders",
    "BernoulliBandit",
    "GaussianBandit",
    "FourRooms",
    "MetaMaze",
    "PointRobot",
    "Reacher",
    "Swimmer",
    "Pong",
    "ProcMaze",
]
