import math
import numpy as np
from mathutils import Matrix
import cv2


HO3D_YCB_LABELS = {
    '002_master_chef_can': 1,
    '003_cracker_box': 2,
    '004_sugar_box': 3,
    '005_tomato_soup_can': 4,
    '006_mustard_bottle': 5,
    '007_tuna_fish_can': 6,
    '008_pudding_box': 7,
    '009_gelatin_box': 8,
    '010_potted_meat_can': 9,
    '011_banana': 10,
    '019_pitcher_base': 11,
    '021_bleach_cleanser': 12,
    '024_bowl': 13,
    '025_mug': 14,
    '035_power_drill': 15,
    '036_wood_block': 16,
    '037_scissors': 17,
    '040_large_marker': 18,
    '051_large_clamp': 19,
    '052_extra_large_clamp': 20,
    '061_foam_brick': 21
}

OBMAN_TO_HO3D_JOINT_INDICES = [0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3, 4, 8, 12, 16, 20]

def get_ho3d_annotation(hand_infos, cam_calib, obj_bounding_box_rest, obj_bounding_box, tf_world_mano, mano_model):
    # swap bounding box indices for blender order -> ho3d order
    obj_bounding_box_rest = np.copy(obj_bounding_box_rest)
    obj_bounding_box = np.copy(obj_bounding_box)

    obj_bounding_box_rest[[2,3]] = obj_bounding_box_rest[[3,2]]
    obj_bounding_box_rest[[6,7]] = obj_bounding_box_rest[[7,6]]

    obj_bounding_box[[2,3]] = obj_bounding_box[[3,2]]
    obj_bounding_box[[6,7]] = obj_bounding_box[[7,6]]

    # convert blender transformation chain to handPose[:3] (rot), handTrans (trans)
    # blender_hand_tf_m = Matrix(tf_world_mano)
    handPose = np.array(hand_infos['hand_pose'])

    tf_mano_wrist = np.eye(4)
    tf_mano_wrist[:3,3] = np.array(mano_model.Jtr.r[0])
    
    tf_mano_wrist_inv = np.eye(4)
    tf_mano_wrist_inv[:3,3] = -np.array(mano_model.Jtr.r[0])


    tf_wrist_grasp = np.eye(4)
    tf_wrist_grasp[:3,:3] = axangle2mat(handPose[:3], np.linalg.norm(handPose[:3]))
    tf_wrist_grasp[:3,3] = np.array(hand_infos['hand_trans'])

    tf_world_grasp = np.dot(tf_world_mano, np.dot(tf_mano_wrist, tf_wrist_grasp))

    tf_wristprime_grasp = np.dot(tf_mano_wrist_inv, tf_world_grasp)


    axis, angle = mat2axangle(np.array(tf_wristprime_grasp[:3,:3]))
    if angle != 0:
        axis = axis*angle

    handPose[:3] = axis
    handTrans = tf_wristprime_grasp[:3,3]

    obj_tf = Matrix(hand_infos['affine_transform'])
    obj_rot = cv2.Rodrigues(np.array(obj_tf.to_3x3()))[0]
    obj_label = HO3D_YCB_LABELS[hand_infos['sample_id']] if hand_infos['sample_id'] in HO3D_YCB_LABELS else -1
    ho3d_anno = {
        'camIDList': ['1'],
        'camMat': cam_calib,
        'handBeta': hand_infos['shape'],
        'handJoints3D': hand_infos['coords_3d'][OBMAN_TO_HO3D_JOINT_INDICES],
        'handPose': handPose,#np.array(hand_infos['hand_pose']),
        'handTrans': handTrans,
        'objCorners3D': obj_bounding_box,
        'objCorners3DRest': obj_bounding_box_rest,
        'objLabel': obj_label,
        'objName': hand_infos['class_id'] + ('_' if hand_infos['class_id'] else '') + hand_infos['sample_id'],
        'objRot': obj_rot,
        'objTrans': np.array(obj_tf.translation)
    }
    return ho3d_anno






def write_depth_img(depth_filename, depth_file_exr):
    """Encode the depth image and write it to file"""
    depth_exr = cv2.imread(depth_file_exr,  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)  
    depth_encoded = np.zeros_like(depth_exr)

    depth_scale = 0.00012498664727900177
    depth = np.uint32(depth_exr[:,:,0] / depth_scale)

    d1 = depth >> 8
    depth_encoded[:, :, 1] = d1
    depth_encoded[:, :, 2] = depth - (d1 << 8)

    cv2.imwrite(depth_filename, depth_encoded)







def axangle2mat(axis, angle, is_normalized=False):
    ''' Rotation matrix for rotation angle `angle` around `axis`

    Parameters
    ----------
    axis : 3 element sequence
       vector specifying axis for rotation.
    angle : scalar
       angle of rotation in radians.
    is_normalized : bool, optional
       True if `axis` is already normalized (has norm of 1).  Default False.

    Returns
    -------
    mat : array shape (3,3)
       rotation matrix for specified rotation

    Notes
    -----
    From: http://en.wikipedia.org/wiki/Rotation_matrix#Axis_and_angle
    '''
    x, y, z = axis
    if not is_normalized:
        n = math.sqrt(x*x + y*y + z*z)
        x = x/n
        y = y/n
        z = z/n
    c = math.cos(angle); s = math.sin(angle); C = 1-c
    xs = x*s;   ys = y*s;   zs = z*s
    xC = x*C;   yC = y*C;   zC = z*C
    xyC = x*yC; yzC = y*zC; zxC = z*xC
    return np.array([
            [ x*xC+c,   xyC-zs,   zxC+ys ],
            [ xyC+zs,   y*yC+c,   yzC-xs ],
            [ zxC-ys,   yzC+xs,   z*zC+c ]])


def mat2axangle(mat, unit_thresh=1e-5):
    """Return axis, angle and point from (3, 3) matrix `mat`

    Parameters
    ----------
    mat : array-like shape (3, 3)
        Rotation matrix
    unit_thresh : float, optional
        Tolerable difference from 1 when testing for unit eigenvalues to
        confirm `mat` is a rotation matrix.

    Returns
    -------
    axis : array shape (3,)
       vector giving axis of rotation
    angle : scalar
       angle of rotation in radians.

    Examples
    --------
    >>> direc = np.random.random(3) - 0.5
    >>> angle = (np.random.random() - 0.5) * (2*math.pi)
    >>> R0 = axangle2mat(direc, angle)
    >>> direc, angle = mat2axangle(R0)
    >>> R1 = axangle2mat(direc, angle)
    >>> np.allclose(R0, R1)
    True

    Notes
    -----
    http://en.wikipedia.org/wiki/Rotation_matrix#Axis_of_a_rotation
    """
    M = np.asarray(mat, dtype=np.float)
    # direction: unit eigenvector of R33 corresponding to eigenvalue of 1
    L, W = np.linalg.eig(M.T)
    i = np.where(np.abs(L - 1.0) < unit_thresh)[0]
    if not len(i):
        raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
    direction = np.real(W[:, i[-1]]).squeeze()
    # rotation angle depending on direction
    cosa = (np.trace(M) - 1.0) / 2.0
    if abs(direction[2]) > 1e-8:
        sina = (M[1, 0] + (cosa-1.0)*direction[0]*direction[1]) / direction[2]
    elif abs(direction[1]) > 1e-8:
        sina = (M[0, 2] + (cosa-1.0)*direction[0]*direction[2]) / direction[1]
    else:
        sina = (M[2, 1] + (cosa-1.0)*direction[1]*direction[2]) / direction[0]
    angle = math.atan2(sina, cosa)
    return direction, angle

