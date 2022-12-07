import sys
sys.path.append('./')

from src.dataFactory.dataFactory import *
from src.common.utilities import * 

if __name__ == '__main__':

    output_path = get_visual_data_path('GAVE_RGB')
    capture_visual_rgb_npys(input_path=GAVE_DATA_PATH, output_path=output_path, class_labels=GAVE_LABELS)

    output_path = get_visual_data_path('GAVE')
    capture_visual_npys(input_path=GAVE_DATA_PATH, output_path=output_path, class_labels=GAVE_LABELS)
    
    output_path = get_visual_data_path('GAVE_AU')
    capture_visual_au_npys(input_path=GAVE_DATA_PATH, output_path=output_path)

    output_path = get_visual_data_path('GAVE_HISTOGRAM_FRAMES_COUNT')
    capture_visual_npys(input_path=GAVE_DATA_PATH, output_path=output_path, class_labels=GAVE_LABELS, selector='histogram_frames_count')

    input_path = get_visual_data_path('GAVE_HISTOGRAM_FRAMES_COUNT')
    output_path = get_visual_data_path('GAVE_AU_HISTOGRAM_FRAMES_COUNT')
    capture_visual_au_npys(input_path=input_path, output_path=output_path)
    
    output_path = get_visual_data_path('GAVE_FSIM_FRAMES_COUNT')
    capture_visual_npys(input_path=GAVE_DATA_PATH, output_path=output_path, class_labels=GAVE_LABELS, selector='fsim_frames_count')

    input_path = get_visual_data_path('GAVE_FSIM_FRAMES_COUNT')
    output_path = get_visual_data_path('GAVE_AU_FSIM_FRAMES_COUNT')
    capture_visual_au_npys(input_path=input_path, output_path=output_path)
    
    output_path = get_visual_data_path('GAVE_SSIM_FRAMES_COUNT')
    capture_visual_npys(input_path=GAVE_DATA_PATH, output_path=output_path, class_labels=GAVE_LABELS, selector='ssim_frames_count')

    input_path = get_visual_data_path('GAVE_SSIM_FRAMES_COUNT')
    output_path = get_visual_data_path('GAVE_AU_SSIM_FRAMES_COUNT')
    capture_visual_au_npys(input_path=input_path, output_path=output_path)
    
    output_path = get_visual_data_path('GAVE_FSM_FRAMES_COUNT')
    capture_visual_npys(input_path=GAVE_DATA_PATH, output_path=output_path, class_labels=GAVE_LABELS, selector='fsm_frames_count')

    input_path = get_visual_data_path('GAVE_FSM_FRAMES_COUNT')
    output_path = get_visual_data_path('GAVE_AU_FSM_FRAMES_COUNT')
    capture_visual_au_npys(input_path=input_path, output_path=output_path)

    output_path = get_visual_data_path('GAVE_HISTOGRAM')
    capture_visual_npys(input_path=GAVE_DATA_PATH, output_path=output_path, class_labels=GAVE_LABELS, selector='histogram')

    input_path = get_visual_data_path('GAVE_HISTOGRAM')
    output_path = get_visual_data_path('GAVE_AU_HISTOGRAM')
    capture_visual_au_npys(input_path=input_path, output_path=output_path)
    
    output_path = get_visual_data_path('GAVE_FSIM')
    capture_visual_npys(input_path=GAVE_DATA_PATH, output_path=output_path, class_labels=GAVE_LABELS, selector='fsim')

    input_path = get_visual_data_path('GAVE_FSIM')
    output_path = get_visual_data_path('GAVE_AU_FSIM')
    capture_visual_au_npys(input_path=input_path, output_path=output_path)
    
    output_path = get_visual_data_path('GAVE_SSIM')
    capture_visual_npys(input_path=GAVE_DATA_PATH, output_path=output_path, class_labels=GAVE_LABELS, selector='ssim')

    input_path = get_visual_data_path('GAVE_SSIM')
    output_path = get_visual_data_path('GAVE_AU_SSIM')
    capture_visual_au_npys(input_path=input_path, output_path=output_path)
    
    output_path = get_visual_data_path('GAVE_FSM')
    capture_visual_npys(input_path=GAVE_DATA_PATH, output_path=output_path, class_labels=GAVE_LABELS, selector='fsm')

    input_path = get_visual_data_path('GAVE_FSM')
    output_path = get_visual_data_path('GAVE_AU_FSM')
    capture_visual_au_npys(input_path=input_path, output_path=output_path)
