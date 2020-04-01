#!/bin/bash
unset PYTHONPATH
results_root=${1:-datageneration/3dnet}
frame_start=${2:-0}
frame_nb=${3:-100}
for i in {{0..100}}
do
    if [ -d $results_root/ho3d/depth ] && [ "$(ls -A $results_root/ho3d/depth)" ]
    then
        frame_start=$(( 1 + 10#$( ls $results_root/ho3d/depth -t | head -n1 | sed 's/\.png//g') ))
    fi
    echo "saving to $results_root starting at $frame_start"
    ../blender-2.78c-linux-glibc219-x86_64/blender -noaudio -b -t 1 -P blender_grasps_sacred.py -- {\
                \"frame_nb\": $frame_nb,\
                \"frame_start\": $frame_start,\
                \"results_root\": \"$results_root\",\
                \"background_datasets\": [\"habitat\"],\
                \"grasp_folder\": \"assets/grasps/3dnet_grasps_final\",\
                \"grasp_split_path\": \"assets/grasps/3dnet_grasps_splits.csv\",\
                \"random_obj_textures\": 1,\
                \"shapenet_root\": \"/media/robotics/Seagate Expansion Drive/lwohlhart_datasets/ShapeNetCore.v2\",\
                \"objects_root\": \"/home/robotics/work/dex-net-new/dex-net/.dexnet/3dnet\",\
                \"obj_models\": \"3dnet\",\
                \"z_max\": 1.2 }
done
