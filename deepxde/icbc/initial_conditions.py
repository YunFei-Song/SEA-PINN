"""Initial conditions."""

__all__ = ["IC"]

import numpy as np

from .boundary_conditions import npfunc_range_autocache
from .. import backend as bkd
from .. import utils


class IC:
    """Initial conditions: y([x, t0]) = func([x, t0])."""

    def __init__(self, geom, func, on_initial, component=0):
        self.geom = geom #几何体
        self.func = npfunc_range_autocache(utils.return_tensor(func))#参考解函数（如ref_sol）
        #将on_initial转换为逐点判断函数
        self.on_initial = lambda x, on: np.array(
            [on_initial(x[i], on[i]) for i in range(len(x))]
        )
        self.component = component

    def filter(self, X):
        #筛选同时满足几何体初始条件且在初始时刻的点
        return X[self.on_initial(X, self.geom.on_initial(X))]

    def collocation_points(self, X):
        return self.filter(X)

    def error(self, X, inputs, outputs, beg, end, aux_var=None):
        #计算神经网络输出和参考解的残差 作为损失项
        values = self.func(X, beg, end, aux_var)#调用ref_sol获取真实值
        if bkd.ndim(values) == 2 and bkd.shape(values)[1] != 1:
            raise RuntimeError(
                "IC function should return an array of shape N by 1 for each component."
                "Use argument 'component' for different output components."
            )
        return outputs[beg:end, self.component : self.component + 1] - values
