import sys
sys.path.append('./')

from src.dataFactory.dataFactory import *
from src.common.utilities import * 

if __name__ == '__main__':

    output_path = get_visual_data_path('BAUM1S_RGB')
    capture_visual_rgb_npys(input_path=BAUM1S_DATA_PATH, output_path=output_path, class_labels=BAUM1S_LABELS)

    output_path = get_visual_data_path('BAUM1S')
    capture_visual_npys(input_path=BAUM1S_DATA_PATH, output_path=output_path, class_labels=BAUM1S_LABELS)
    
    output_path = get_visual_data_path('BAUM1S_AU')
    capture_visual_au_npys(input_path=BAUM1S_DATA_PATH, output_path=output_path)

    output_path = get_visual_data_path('BAUM1S_HISTOGRAM_FRAMES_COUNT')
    capture_visual_npys(input_path=BAUM1S_DATA_PATH, output_path=output_path, class_labels=BAUM1S_LABELS, selector='histogram_frames_count')

    input_path = get_visual_data_path('BAUM1S_HISTOGRAM_FRAMES_COUNT')
    output_path = get_visual_data_path('BAUM1S_AU_HISTOGRAM_FRAMES_COUNT')
    capture_visual_au_npys(input_path=input_path, output_path=output_path)
    
    output_path = get_visual_data_path('BAUM1S_FSIM_FRAMES_COUNT')
    capture_visual_npys(input_path=BAUM1S_DATA_PATH, output_path=output_path, class_labels=BAUM1S_LABELS, selector='fsim_frames_count')

    input_path = get_visual_data_path('BAUM1S_FSIM_FRAMES_COUNT')
    output_path = get_visual_data_path('BAUM1S_AU_FSIM_FRAMES_COUNT')
    capture_visual_au_npys(input_path=input_path, output_path=output_path)
    
    output_path = get_visual_data_path('BAUM1S_SSIM_FRAMES_COUNT')
    capture_visual_npys(input_path=BAUM1S_DATA_PATH, output_path=output_path, class_labels=BAUM1S_LABELS, selector='ssim_frames_count')

    input_path = get_visual_data_path('BAUM1S_SSIM_FRAMES_COUNT')
    output_path = get_visual_data_path('BAUM1S_AU_SSIM_FRAMES_COUNT')
    capture_visual_au_npys(input_path=input_path, output_path=output_path)
    
    output_path = get_visual_data_path('BAUM1S_FSM_FRAMES_COUNT')
    capture_visual_npys(input_path=BAUM1S_DATA_PATH, output_path=output_path, class_labels=BAUM1S_LABELS, selector='fsm_frames_count')

    input_path = get_visual_data_path('BAUM1S_FSM_FRAMES_COUNT')
    output_path = get_visual_data_path('BAUM1S_AU_FSM_FRAMES_COUNT')
    capture_visual_au_npys(input_path=input_path, output_path=output_path)

    output_path = get_visual_data_path('BAUM1S_HISTOGRAM')
    capture_visual_npys(input_path=BAUM1S_DATA_PATH, output_path=output_path, class_labels=BAUM1S_LABELS, selector='histogram')

    input_path = get_visual_data_path('BAUM1S_HISTOGRAM')
    output_path = get_visual_data_path('BAUM1S_AU_HISTOGRAM')
    capture_visual_au_npys(input_path=input_path, output_path=output_path)
    
    output_path = get_visual_data_path('BAUM1S_FSIM')
    capture_visual_npys(input_path=BAUM1S_DATA_PATH, output_path=output_path, class_labels=BAUM1S_LABELS, selector='fsim')

    input_path = get_visual_data_path('BAUM1S_FSIM')
    output_path = get_visual_data_path('BAUM1S_AU_FSIM')
    capture_visual_au_npys(input_path=input_path, output_path=output_path)
    
    output_path = get_visual_data_path('BAUM1S_SSIM')
    capture_visual_npys(input_path=BAUM1S_DATA_PATH, output_path=output_path, class_labels=BAUM1S_LABELS, selector='ssim')

    input_path = get_visual_data_path('BAUM1S_SSIM')
    output_path = get_visual_data_path('BAUM1S_AU_SSIM')
    capture_visual_au_npys(input_path=input_path, output_path=output_path)
    
    output_path = get_visual_data_path('BAUM1S_FSM')
    capture_visual_npys(input_path=BAUM1S_DATA_PATH, output_path=output_path, class_labels=BAUM1S_LABELS, selector='fsm')

    input_path = get_visual_data_path('BAUM1S_FSM')
    output_path = get_visual_data_path('BAUM1S_AU_FSM')
    capture_visual_au_npys(input_path=input_path, output_path=output_path)
