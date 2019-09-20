import configargparse
import cv2
import pandas as pd
from collections import defaultdict
import smplx
import torch
from src.utils import JointMap
import os
import numpy as np
import trimesh
from src.camera import create_camera
import pyrender
from src.fit_script import FittingMonitor

# initializations
model_params = {
    'joint_mapper': JointMap(),
    'create_body_pose': False
}
DEVICE = torch.device('cuda')
DTYPE = torch.float32


with FittingMonitor(batch_size=1, visualize=True, **{}) as monitor:
    pass


def parse_config():
    arg_formatter = configargparse.ArgumentDefaultsHelpFormatter

    cfg_parser = configargparse.YAMLConfigFileParser
    description = 'Visualization demo'
    parser = configargparse.ArgParser(formatter_class=arg_formatter,
                                      config_file_parser_class=cfg_parser,
                                      description=description,
                                      prog='Visualization Demo')
    parser.add_argument('--image_path',
                        default='./input/img.png',
                        help='Path of the image for printing the model.')
    parser.add_argument('--smpl_params_path',
                        default='./input/img_3d_params.pkl',
                        help='Path to 3D parameters of the human body.')
    parser.add_argument('--output_folder',
                        default='./output',
                        help='Path to the folder for the 3d model to be rendered.')
    parser.add_argument('--smpl_model_folder',
                        default='',
                        help='Path to the downloadable SMPL models folder.')

    args = parser.parse_args()
    args_dict = vars(args)
    return args_dict


def get_model_params_from_file(pickel_path):
    pickl_file = pd.read_pickle(pickel_path)
    body_pose = pickl_file['body_pose']
    body_shape = pickl_file['betas']
    global_orient = pickl_file['global_orient']
    camera_translation = pickl_file['camera_translation']
    return body_pose, body_shape, global_orient, camera_translation


def render_3d_model_from_params(body_pose, body_shape, global_orient, camera_translation, body_model, img):
    new_params = defaultdict(global_orient=global_orient,
                             betas=body_shape)

    body_model.reset_params(**new_params)

    model_output = body_model(return_verts=True, body_pose=body_pose)
    vertices = model_output.vertices.detach().cpu().numpy().squeeze()

    out_mesh = trimesh.Trimesh(vertices, body_model.faces)
    rot = trimesh.transformations.rotation_matrix(
        np.radians(180), [1, 0, 0])
    out_mesh.apply_transform(rot)

    focal_length = 5000.0
    camera = create_camera(focal_length_x_input=focal_length,
                           focal_length_y_input=focal_length,
                           dtype=DTYPE,
                           **{})
    camera = camera.to(device=DEVICE)

    # Create the camera object
    H, W, _ = img.shape
    camera.translation[:] = torch.as_tensor(camera_translation).view_as(camera.translation)
    camera.center[:] = torch.tensor([W, H], dtype=DTYPE) * 0.5

    baseColorFactor = (1.0, 1.0, 0.95, 1.0)

    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        alphaMode='OPAQUE',
        baseColorFactor=baseColorFactor)
    mesh = pyrender.Mesh.from_trimesh(
        out_mesh,
        material=material)

    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                           ambient_light=(0.3, 0.3, 0.3))
    scene.add(mesh, 'mesh')

    camera_center = camera.center.detach().cpu().numpy().squeeze()
    camera_transl = camera.translation.detach().cpu().numpy().squeeze()
    # Equivalent to 180 degrees around the y-axis. Transforms the fit to
    # OpenGL compatible coordinate system.
    camera_transl[0] *= -1.0

    camera_pose = np.eye(4)
    camera_pose[:3, 3] = camera_transl

    camera = pyrender.camera.IntrinsicsCamera(
        fx=focal_length, fy=focal_length,
        cx=camera_center[0], cy=camera_center[1])
    scene.add(camera, pose=camera_pose)

    # Get the lights from the viewer
    light_nodes = monitor.mv.viewer._create_raymond_lights()
    for node in light_nodes:
        scene.add_node(node)

    r = pyrender.OffscreenRenderer(viewport_width=W,
                                   viewport_height=H,
                                   point_size=1.0)

    color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
    color = color.astype(np.float32) / 255.0

    valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]

    output_img = 255 - ((255 - color[:, :, :-1]) * valid_mask +
                        (1 - valid_mask) * img)

    output_img = (output_img * 255).astype(np.uint8)

    return output_img


def create_smpl_model(**args):
    assert os.path.exists(
        args['smpl_model_folder']), 'Path {} does not exist in argument smpl_model_folder!'.format(
        args['smpl_model_folder'])

    model_params['model_path'] = args['smpl_model_folder']
    body_model = smplx.create(model_type='smpl', gender='male', **model_params)
    body_model = body_model.to(device=DEVICE)
    return body_model


def main(**args):
    img = cv2.imread(args['image_path'])
    body_pose, body_shape, global_orient, camera_translation = get_model_params_from_file(
        args['smpl_params_path'])
    body_model = create_smpl_model(**args)
    rendered_image = render_3d_model_from_params(body_pose, body_shape, global_orient,
                                                 camera_translation, body_model,
                                                 img)
    cv2.imwrite(os.path.join(args['output_folder'], os.path.basename(args['image_path'])), rendered_image)

    pass


def understanding_3d_model_params_module(**args):

    img = cv2.imread(args['image_path'])
    body_pose, body_shape, global_orient, camera_translation = get_model_params_from_file(
        args['smpl_params_path'])
    body_model = create_smpl_model(**args)

    # Choose param to change - change this parameter also in the loop ahead
    param_for_change = body_pose

    # Choose index to change in the param
    ind = 0

    assert 0 <= ind < len(param_for_change), 'Index ind = {} is not in the range of param_for_change = {}'.format(
        ind, param_for_change)

    # Choose range for change of the param
    values_range = np.linspace(-0.5, 0.5, 7)

    images = np.zeros(shape=(len(values_range), img.shape[0], img.shape[1], img.shape[2]))

    for ii in range(len(values_range)):
        param_for_change[0, ind] = values_range[ii]

        images[ii, ...] = (render_3d_model_from_params(param_for_change, body_shape, global_orient,
                                                     camera_translation, body_model,
                                                     img)).astype(np.float)

    new_image = (np.mean(images, axis=0)).astype(np.uint)

    cv2.imwrite(os.path.join(args['output_folder'], os.path.basename(args['image_path'])), new_image)

    pass


if __name__ == "__main__":
    args = parse_config()
    main(**args)
    # understanding_3d_model_params_module(**args)
    pass
