
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from collections import namedtuple

import torch
import torch.nn as nn

from smplx.lbs import transform_mat

PerspParams = namedtuple('ModelOutput',
                         ['rotation', 'translation', 'center',
                          'focal_length'])


def create_camera(type_of_camera='persp', **kwargs):
    if type_of_camera.lower() == 'persp':
        return Camera(**kwargs)
    else:
        raise ValueError('Uknown camera type: {}'.format(type_of_camera))


class Camera(nn.Module):

    FOCAL_LENGTH = 5000

    def __init__(self, camera_rotation=None, camera_translation=None,
                 focal_length_x_input=None, focal_length_y_input=None,
                 batch_size=1,
                 center=None, dtype=torch.float32, **kwargs):
        super(Camera, self).__init__()
        self.batch_size = batch_size
        self.dtype = dtype

        self.register_buffer('zero',
                             torch.zeros([batch_size], dtype=dtype))

        if focal_length_x_input is None or type(focal_length_x_input) == float:
            focal_length_x_input = torch.full(
                [batch_size],
                self.FOCAL_LENGTH if focal_length_x_input is None else
                focal_length_x_input,
                dtype=dtype)

        if focal_length_y_input is None or type(focal_length_y_input) == float:
            focal_length_y_input = torch.full(
                [batch_size],
                self.FOCAL_LENGTH if focal_length_y_input is None else
                focal_length_y_input,
                dtype=dtype)

        self.register_buffer('focal_length_x', focal_length_x_input)
        self.register_buffer('focal_length_y', focal_length_y_input)

        if center is None:
            center = torch.zeros([batch_size, 2], dtype=dtype)
        self.register_buffer('center', center)

        if camera_rotation is None:
            camera_rotation = torch.eye(
                3, dtype=dtype).unsqueeze(dim=0).repeat(batch_size, 1, 1)

        camera_rotation = nn.Parameter(camera_rotation, requires_grad=True)
        self.register_parameter('rotation', camera_rotation)

        if camera_translation is None:
            camera_translation = torch.zeros([batch_size, 3], dtype=dtype)

        camera_translation = nn.Parameter(camera_translation,
                                          requires_grad=True)
        self.register_parameter('translation', camera_translation)

    def forward(self, camera_points):
        device = camera_points.device

        with torch.no_grad():
            camera_mat = torch.zeros([self.batch_size, 2, 2],
                                     dtype=self.dtype, device=camera_points.device)
            camera_mat[:, 0, 0] = self.focal_length_x
            camera_mat[:, 1, 1] = self.focal_length_y

        camera_transform = transform_mat(self.rotation,
                                         self.translation.unsqueeze(dim=-1))
        homog_coord = torch.ones(list(camera_points.shape)[:-1] + [1],
                                 dtype=camera_points.dtype,
                                 device=device)
        # Convert the points to homogeneous coordinates
        points_h = torch.cat([camera_points, homog_coord], dim=-1)

        projected_points = torch.einsum('bki,bji->bjk',
                                        [camera_transform, points_h])

        img_points = torch.div(projected_points[:, :, :2],
                               projected_points[:, :, 2].unsqueeze(dim=-1))
        img_points = torch.einsum('bki,bji->bjk', [camera_mat, img_points]) \
            + self.center.unsqueeze(dim=1)
        return img_points