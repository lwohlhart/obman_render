import os
import uuid
import numpy as np

import bpy


def get_all_backgrounds(folder='/sequoia/data2/gvarol/datasets/LSUN/data/img'):
    train_list_path = os.path.join(folder, 'train_img.txt')
    with open(train_list_path) as f:
        lines = f.readlines()
    bg_names = [os.path.join(folder, line.strip()) for line in lines]
    return bg_names

def set_depth_occlusion_nodes(node_tree, render_node):
    hand_thresh = node_tree.nodes.new(type="CompositorNodeMath")
    hand_thresh.name = 'hand_thresh'
    hand_thresh.location = -600, 800
    hand_thresh.operation = 'LESS_THAN'
    hand_thresh.inputs[1].default_value = 100.0
    node_tree.links.new(render_node.outputs['Z'], hand_thresh.inputs[0])

    hand_erode = node_tree.nodes.new(type='CompositorNodeDilateErode')
    hand_erode.name = 'hand_erode'
    hand_erode.location = -500, 800
    hand_erode.mode = 'FEATHER'
    hand_erode.distance = 20
    hand_erode.falloff = 'LINEAR'
    node_tree.links.new(hand_thresh.outputs[0], hand_erode.inputs[0])

    occlusion_region = node_tree.nodes.new(type="CompositorNodeMath")
    occlusion_region.name = 'occlusion_region'
    occlusion_region.location = -400, 800
    occlusion_region.operation = 'SUBTRACT'
    occlusion_region.inputs[1].default_value = 100.0
    node_tree.links.new(hand_erode.outputs[0], occlusion_region.inputs[0])
    node_tree.links.new(hand_thresh.outputs[0], occlusion_region.inputs[1])

    occlusion_random_noise = node_tree.nodes.new(type="CompositorNodeTexture")
    occlusion_random_noise.name = 'occlusion_random_noise'
    occlusion_random_noise.location = -300, 800

    occlusion_noise_texture = bpy.data.textures.get('occlusion_noise')
    if occlusion_noise_texture is None:
        occlusion_noise_texture = bpy.data.textures.new(name="occlusion_noise", type='DISTORTED_NOISE')
    occlusion_noise_texture.distortion = 3.0 + np.random.random()
    occlusion_random_noise.texture = occlusion_noise_texture
    node_tree.links.new(occlusion_region.outputs[0], occlusion_random_noise.inputs[0])

    occlusion_random_noise_blur = node_tree.nodes.new(type="CompositorNodeBlur")
    occlusion_random_noise_blur.name = 'occlusion_random_noise_blur'
    occlusion_random_noise_blur.location = -200, 800
    occlusion_random_noise_blur.filter_type = 'CUBIC' # not really necessary
    occlusion_random_noise_blur.size_x = 5.0
    occlusion_random_noise_blur.size_y = 5.0
    node_tree.links.new(occlusion_random_noise.outputs[0], occlusion_random_noise_blur.inputs[0])

    occlusion_region_fused = node_tree.nodes.new(type="CompositorNodeMath")
    occlusion_region_fused.name = 'occlusion_region_fused'
    occlusion_region_fused.location = -100, 800
    occlusion_region_fused.operation = 'MULTIPLY'
    node_tree.links.new(occlusion_region.outputs[0], occlusion_region_fused.inputs[0])
    node_tree.links.new(occlusion_random_noise_blur.outputs[0], occlusion_region_fused.inputs[1])

    occlusion_region_thresh = node_tree.nodes.new(type="CompositorNodeMath")
    occlusion_region_thresh.name = 'occlusion_region_thresh'
    occlusion_region_thresh.location = 0, 800
    occlusion_region_thresh.operation = 'GREATER_THAN'
    occlusion_region_thresh.inputs[1].default_value = 0.15
    node_tree.links.new(occlusion_region_fused.outputs[0], occlusion_region_thresh.inputs[0])

    ## occlusion_region_thresh actually produces the depth occlusion mask


    occlusion_map_depth = node_tree.nodes.new(type="CompositorNodeMapRange")
    occlusion_map_depth.name = 'occlusion_map_depth'
    occlusion_map_depth.location = 100, 800
    occlusion_map_depth.inputs[1].default_value = 0.0
    occlusion_map_depth.inputs[2].default_value = 1.0
    occlusion_map_depth.inputs[3].default_value = 1.0
    occlusion_map_depth.inputs[4].default_value = 1000.0
    node_tree.links.new(occlusion_region_thresh.outputs[0], occlusion_map_depth.inputs[0])

    return occlusion_region_thresh, occlusion_map_depth


def set_cycle_nodes(scene, bg_name, segm_path, depth_path, bg_depth_name=None):
    # Get node tree
    scene.use_nodes = True
    scene.render.layers['RenderLayer'].use_pass_material_index = True
    scene.render.alpha_mode = 'TRANSPARENT'
    node_tree = scene.node_tree

    # Remove existing nodes
    for n in node_tree.nodes:
        node_tree.nodes.remove(n)

    # Get Z combination node
    zcombined_node = node_tree.nodes.new(type='CompositorNodeZcombine')
    zcombined_node.location = 0, 100

    # Get rendering
    render_node = node_tree.nodes.new(type="CompositorNodeRLayers")
    render_node.location = -400, -200

    # Get background
    if bg_name is not None:
        bg_img = bpy.data.images.load(bg_name)
        bg_node = node_tree.nodes.new(type="CompositorNodeImage")

        bg_node.image = bg_img
        bg_node.location = -400, 200

        # Scale background by croping
        bg_scale_node = node_tree.nodes.new(type="CompositorNodeScale")
        bg_scale_node.space = "RENDER_SIZE"
        bg_scale_node.frame_method = "CROP"
        bg_scale_node.location = -200, 200
        node_tree.links.new(bg_node.outputs[0], bg_scale_node.inputs[0])
        node_tree.links.new(bg_scale_node.outputs[0], zcombined_node.inputs[2])
        zcombined_node.inputs[3].default_value = 10.0 # default background depth far

        bg_depth_img = None
        if bg_depth_name is not None and os.path.exists(bg_depth_name):
            bg_depth_img = bpy.data.images.load(bg_depth_name)
            bg_depth_node = node_tree.nodes.new(type="CompositorNodeImage")
            bg_depth_node.image = bg_depth_img
            bg_depth_node.location = -600, 500

            bg_depth_adjust_node = node_tree.nodes.new(type='CompositorNodeMapValue')
            bg_depth_adjust_node.size[0] = 1 # 100 # depth scale
            bg_depth_adjust_node.offset[0] = 1 # maybe send back with offset
            bg_depth_adjust_node.location = -400, 500
            node_tree.links.new(bg_depth_node.outputs[0], bg_depth_adjust_node.inputs[0])

            bg_depth_scale_node = node_tree.nodes.new(type="CompositorNodeScale")
            bg_depth_scale_node.space = "RENDER_SIZE"
            bg_depth_scale_node.frame_method = "CROP"
            bg_depth_scale_node.location = -200, 500

            node_tree.links.new(bg_depth_adjust_node.outputs[0], bg_depth_scale_node.inputs[0])
            if False: # actually check if we want to mask occlusions
                node_tree.links.new(bg_depth_scale_node.outputs[0], zcombined_node.inputs[3])
            else:
                _, occlusion_map_depth = set_depth_occlusion_nodes(node_tree, render_node)

                occlusion_multiplier = node_tree.nodes.new(type="CompositorNodeMath")
                occlusion_multiplier.name = 'occlusion_multiplier'
                occlusion_multiplier.location = -100, 500
                occlusion_multiplier.operation = 'MULTIPLY'

                node_tree.links.new(bg_depth_scale_node.outputs[0], occlusion_multiplier.inputs[0])
                node_tree.links.new(occlusion_map_depth.outputs[0], occlusion_multiplier.inputs[1])
                node_tree.links.new(occlusion_multiplier.outputs[0], zcombined_node.inputs[3])



    # Get Z pass
    depth_node =  node_tree.nodes.new('CompositorNodeOutputFile')
    depth_node.format.file_format = 'OPEN_EXR'
    depth_node.base_path = os.path.dirname(depth_path)
    depth_node.file_slots[0].path = os.path.basename(depth_path)
    depth_node.location = 200, 0
    node_tree.links.new(render_node.outputs['Z'], zcombined_node.inputs[1])
    node_tree.links.new(zcombined_node.outputs['Z'], depth_node.inputs[0])

    node_tree.links.new(render_node.outputs[0], zcombined_node.inputs[0])

    # # Overlay background image and rendering
    # alpha_node = node_tree.nodes.new(type="CompositorNodeAlphaOver")
    # alpha_node.location = 0, 200
    # node_tree.links.new(scale_node.outputs[0], alpha_node.inputs[1])
    # node_tree.links.new(render_node.outputs[0], alpha_node.inputs[2])

    comp_node = node_tree.nodes.new(type="CompositorNodeComposite")
    comp_node.location = 200, 200
    node_tree.links.new(zcombined_node.outputs[0], comp_node.inputs[0])

    # Add segmentation node
    scale_node = node_tree.nodes.new(type='CompositorNodeMapRange')
    scale_node.location = 0, -200
    scale_node.inputs[1].default_value = 0
    scale_node.inputs[2].default_value = 255
    # left_handbase_mask.index = 21 # left_finger_mask.index = 23
    segm_view = node_tree.nodes.new(type="CompositorNodeOutputFile")
    segm_view.location = 200, -200
    segm_view.format.file_format = 'PNG'
    segm_view.base_path = segm_path
    temp_filename = uuid.uuid4().hex
    segm_view.file_slots[0].path = temp_filename
    node_tree.links.new(render_node.outputs['IndexMA'], scale_node.inputs[0])
    node_tree.links.new(scale_node.outputs[0], segm_view.inputs[0])
    tmp_segm_file = os.path.join(segm_path, '{}0001.png'.format(temp_filename))

    return tmp_segm_file
