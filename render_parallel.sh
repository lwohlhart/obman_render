#!/bin/bash

tmux new-session -d -s oaho_render_1 './render_grasps.sh datageneration/3dnet_seq_10000000 10000000'
tmux new-session -d -s oaho_render_2 './render_grasps.sh datageneration/3dnet_seq_20000000 20000000'
tmux new-session -d -s oaho_render_3 './render_grasps.sh datageneration/3dnet_seq_30000000 30000000'
tmux new-session -d -s oaho_render_4 './render_grasps.sh datageneration/3dnet_seq_40000000 40000000'
# tmux new-session -d -s oaho_render_5 './render_grasps.sh datageneration/3dnet_seq_50000000 50000000'
# tmux new-session -d -s oaho_render_6 './render_grasps.sh datageneration/3dnet_seq_60000000 60000000'
