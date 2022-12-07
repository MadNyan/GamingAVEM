import sys
sys.path.append('./')

from src.dataPreprocess.preprocessVisual import capture_visual_rgb_npys as rgb
from src.dataPreprocess.preprocessVisual import capture_visual_npys as npy
from src.dataPreprocess.preprocessVisual import capture_visual_au_npys as au
from src.common.utilities import * 

def capture_visual_rgb_npys(input_path=ENTERFACE_DATA_PATH, output_path=ENTERFACE_VISUAL_NPY_DATA_PATH, class_labels=ENTERFACE_LABELS, time_F=1, resize_x=224, resize_y=224):
    timepoint_start, _ = get_timepoint()

    rgb(input_path=input_path, output_path=output_path, class_labels=class_labels)

    timepoint_end, _ = get_timepoint()
    _, time_cost_msg = get_time_cost(timepoint_start, timepoint_end)    
    write_result(output_path + '/time_cost.txt', 'Training_time_cost: {}'.format(time_cost_msg))

def capture_visual_npys(input_path=ENTERFACE_DATA_PATH, output_path=ENTERFACE_VISUAL_NPY_DATA_PATH, class_labels=ENTERFACE_LABELS, time_F=1,
                        resize_x=224, resize_y=224, to_local_binary_pattern=False, to_interlaced_derivative_pattern=False, selector='',is_reverse=False , key_frames_count=0):
    timepoint_start, _ = get_timepoint()

    npy(input_path=input_path, output_path=output_path, class_labels=class_labels, selector=selector)

    timepoint_end, _ = get_timepoint()
    _, time_cost_msg = get_time_cost(timepoint_start, timepoint_end)    
    write_result(output_path + '/time_cost.txt', 'Training_time_cost: {}'.format(time_cost_msg))

def capture_visual_au_npys(input_path=ENTERFACE_DATA_PATH, output_path=ENTERFACE_VISUAL_AU_DATA_PATH):
    timepoint_start, _ = get_timepoint()

    au(input_path=input_path, output_path=output_path)

    timepoint_end, _ = get_timepoint()
    _, time_cost_msg = get_time_cost(timepoint_start, timepoint_end)    
    write_result(output_path + '/time_cost.txt', 'Training_time_cost: {}'.format(time_cost_msg))