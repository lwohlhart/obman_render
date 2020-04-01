import os
import random
import pickle
import sys
import argparse

# from sacred import Experiment
import cv2
import numpy as np
import open3d as o3d

root = '.'
sys.path.insert(0, root)
mano_path = os.environ.get('MANO_LOCATION', None)
if mano_path is None:
    raise ValueError('Environment variable MANO_LOCATION not defined'
                     'Please follow the README.md instructions')
sys.path.insert(0, os.path.join(mano_path, 'webuser'))

# from obman_render.grasps.grasputils import read_grasp_folder
from obman_render import mesh_manip
# from obman_render.blenderobj import load_obj_model, delete_obj_model
# from serialization import load_model
from smpl_handpca_wrapper import load_model as smplh_load_model
# from obman_render import ho3dutils


# ex = Experiment('select_cmu_poses')

# @ex.config
# def exp_config():
    


named_index = { name: index for index,name in enumerate(['root',
'hip_l',
'hip_r',
'spine_lower',
'knee_l',
'knee_r',
'spine_middle',
'ankle_l',
'ankle_r',
'chest',
'foot_l',
'foot_r',
'neck',
'collar_l',
'collar_r',
'head_',
'shoulder_l',
'shoulder_r',
'elbow_l',
'elbow_r',
'wrist_l',
'wrist_r'])}


# @ex.automain
def run(smpl_data_path, smpl_model_path, mano_right_path):
    parser = argparse.ArgumentParser(description='cmu pose selection')    
    parser.add_argument('-o', '--file-out', type=str, default='good_cmu_poses_test.txt', help="list of prepared object versions")
    parser.add_argument('-s', '--start', type=int, default=0)
    parser.add_argument('-e', '--end', type=int, default=5338)
    parser.add_argument('-v', '--vis', action='store_true')
    args = parser.parse_args()

    # Load SMPL+H model
    ncomps = 45
    smplh_model = smplh_load_model(smpl_model_path, ncomps=2 * ncomps, flat_hand_mean=True)
    # Load smpl info
    smpl_data = np.load(smpl_data_path)

    smplh_verts, faces = smplh_model.r, smplh_model.f

    if args.vis:
        visualizer = o3d.visualization.Visualizer()
        visualizer.create_window()

        mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(smplh_verts.copy()),
            triangles=o3d.utility.Vector3iVector(faces.copy())
        )
        visualizer.add_geometry(mesh)

        coord_frames = {}
        COORD_FRAME_POINTS = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]]) * 0.5
        def add_coord_frame():
            coord_frame = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(COORD_FRAME_POINTS),
                lines=o3d.utility.Vector2iVector([[0,1],[0,2],[0,3]]),
            )
            coord_frame.colors = o3d.utility.Vector3dVector([[1,0,0], [0,1,0], [0,0,1]])
            return coord_frame

        def update_coord_frame(frame, tf):
            p = (tf.dot(np.vstack([COORD_FRAME_POINTS.transpose(), np.ones((4))]))[:3,:]).transpose()
            frame.points = o3d.utility.Vector3dVector(p)
        
        coord_frames['origin'] = add_coord_frame()
        coord_frames['wrist'] = add_coord_frame()
        ro = visualizer.get_render_option()
        ro.background_color = (1,1,1)
        ro.mesh_show_wireframe = True
        visualizer.add_geometry(coord_frames['origin'])
        visualizer.add_geometry(coord_frames['wrist'])

        def vis(model):
            # visualizer.clear_geometries()
            mesh.vertices = o3d.utility.Vector3dVector(model.r.copy())
            # mesh.vertices = o3d.utility.Vector3dVector(model.r.copy())                
            update_coord_frame(coord_frames['wrist'], model.A_global[21])

            visualizer.update_geometry(coord_frames['wrist'])
            visualizer.update_geometry(mesh)
            visualizer.poll_events()
            visualizer.update_renderer()

    f = open(args.file_out, 'w')
    for cmu_idx in range(args.start, args.end):#[0,1,2,3]:
        cmu_parms, fshapes, name = mesh_manip.load_body_data(smpl_data, idx=cmu_idx)
        pose_data = cmu_parms[name]['poses']
        nframes = pose_data.shape[0]
        for frame in range(nframes):


            pose = np.zeros(smplh_model.pose.size)            
            body_idx = 72
            pose[:body_idx] = pose_data[frame]
            pose[0:3] = [-np.pi/2, 0, 0]

            smplh_model.pose[:] = pose

                # o3d.visualization.draw_geometries([mesh])
            joints = smplh_model.J_transformed.r

            root = joints[named_index['root']]
            wrist_r = joints[named_index['wrist_r']]
            elbow_r = joints[named_index['elbow_r']]

            root_to_wrist_r = wrist_r - root 
            elbow_r_to_wrist_r = wrist_r - elbow_r
            # print((root_to_wrist_r, elbow_r_to_wrist_r))
            # print ('---- 21 ----')
            # print(np.dot(smplh_model.A_global[21][:3,:3], np.array([0,0,1])) )
            # print(np.dot(smplh_model.A_global[21][:3,:3], np.array([0,1,0])) )
            # print(np.dot(smplh_model.A_global[21][:3,:3], np.array([1,0,0])) )

            # print ('---- 20 ----')
            # print(np.dot(smplh_model.A_global[20][:3,:3], np.array([0,0,1])) )
            # print(np.dot(smplh_model.A_global[20][:3,:3], np.array([0,1,0])) )
            # print(np.dot(smplh_model.A_global[20][:3,:3], np.array([1,0,0])) )

            wrist_r_rot = np.array(smplh_model.A_global[21][:3,:3])
            vectors = np.array([[0, -1, 0],
                                [-0.8, -1, 0],
                                [-0.8, -1, -0.5],
                                [-0.8, -1, +0.5]])
            vectors = vectors / np.linalg.norm(vectors, axis=1)[:,None]
            vectors = vectors.dot(wrist_r_rot)
            global_y_angles = np.arccos(np.clip(vectors[:,1], -1.0, 1.0))
            
            if np.linalg.norm(root_to_wrist_r) > 0.25 and np.any(np.abs(global_y_angles) < np.pi/4) and elbow_r_to_wrist_r[1] > 0:
                print((cmu_idx, frame))
                f.write('%i %i\n' % (cmu_idx, frame))
                if args.vis:
                    ro.background_color = (0.8,1,0.8)
            else:
                # print('{} {} {}'.format(np.linalg.norm(root_to_wrist_r),  root_to_wrist_r[1], elbow_r_to_wrist_r[1]))
                if args.vis:
                    ro.background_color = (1,0.8,0.8)
                
            if args.vis:
                vis(smplh_model)
            #print(global_y_angles)
            
    f.close()

if __name__ == '__main__':
    smpl_data_path = 'assets/SURREAL/smpl_data/smpl_data.npz'
    mano_path = 'assets'
    smpl_model_path = os.path.join(mano_path, 'models', 'SMPLH_female.pkl')
    mano_right_path = os.path.join(mano_path, 'models', 'MANO_RIGHT.pkl')

    run(smpl_data_path, smpl_model_path, mano_right_path)
