import sys
sys.path.append('./')

from src.dataPreprocess.preprocessVisual import *
from src.common.utilities import * 

if __name__ == '__main__':

    output_path = get_visual_data_path('BAUM1A_AU')
    capture_visual_au_npys(input_path=BAUM1A_DATA_PATH, output_path=output_path)

    output_path = get_visual_data_path('BAUM1A_RGB')
    capture_visual_rgb_npys(input_path=BAUM1A_DATA_PATH, output_path=output_path, class_labels=BAUM1A_LABELS)

    #output_path = get_visual_data_path('BAUM1A')
    #capture_visual_npys(input_path=BAUM1A_DATA_PATH, output_path=output_path, class_labels=BAUM1A_LABELS)

    #output_path = get_visual_data_path('BAUM1A_HISTOGRAM_FRAMES_COUNT')
    #capture_visual_npys(input_path=BAUM1A_DATA_PATH, output_path=output_path, class_labels=BAUM1A_LABELS, selector='histogram_frames_count')
    
    output_path = get_visual_data_path('BAUM1A_FSIM_FRAMES_COUNT')
    capture_visual_npys(input_path=BAUM1A_DATA_PATH, output_path=output_path, class_labels=BAUM1A_LABELS, selector='fsim_frames_count')
    
    #output_path = get_visual_data_path('BAUM1A_SSIM_FRAMES_COUNT')
    #capture_visual_npys(input_path=BAUM1A_DATA_PATH, output_path=output_path, class_labels=BAUM1A_LABELS, selector='ssim_frames_count')
    
    #output_path = get_visual_data_path('BAUM1A_FSM_FRAMES_COUNT')
    #capture_visual_npys(input_path=BAUM1A_DATA_PATH, output_path=output_path, class_labels=BAUM1A_LABELS, selector='fsm_frames_count')

    #output_path = get_visual_data_path('BAUM1A_HISTOGRAM')
    #capture_visual_npys(input_path=BAUM1A_DATA_PATH, output_path=output_path, class_labels=BAUM1A_LABELS, selector='histogram')
    
    #output_path = get_visual_data_path('BAUM1A_FSIM')
    #capture_visual_npys(input_path=BAUM1A_DATA_PATH, output_path=output_path, class_labels=BAUM1A_LABELS, selector='fsim')
    
    #output_path = get_visual_data_path('BAUM1A_SSIM')
    #capture_visual_npys(input_path=BAUM1A_DATA_PATH, output_path=output_path, class_labels=BAUM1A_LABELS, selector='ssim')
    
    #output_path = get_visual_data_path('BAUM1A_FSM')
    #capture_visual_npys(input_path=BAUM1A_DATA_PATH, output_path=output_path, class_labels=BAUM1A_LABELS, selector='fsm')