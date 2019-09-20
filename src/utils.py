
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

import torch
import torch.nn as nn

class JointMap(nn.Module):
    def __init__(self, joints=None):
        super(JointMap, self).__init__()
        if joints is None:
            self.joints = joints
        else:
            self.register_buffer('joint_maps', torch.tensor(joints, dtype=torch.long))

    def forward(self, joints, **kwargs):
        if self.joints is None:
            return joints
        else:
            return   torch.index_select(joints, 1, self.joints)


def to_tensor(tensor, dtype=torch.float32):
    if torch.Tensor == type(tensor):
        return tensor.clone().detach()
    else:
        return torch.tensor(tensor, dtype)


def rel_change(prev_val, curr_val):
    return (prev_val - curr_val) / max([np.abs(prev_val), np.abs(curr_val), 1])


class GMoF(nn.Module):
    def __init__(self, rho=1):
        super(GMoF, self).__init__()
        self.rho = rho

    def extra_repr(self):
        return 'rho = {}'.format(self.rho)

    def forward(self, residual):
        squared_res = residual ** 2
        dist = torch.div(squared_res, squared_res + self.rho ** 2)
        return self.rho ** 2 * dist