import sys
sys.path.append('./')

from src.dataPreprocess.preprocessVisual import *
from src.common.utilities import * 

if __name__ == '__main__':

    output_path = get_visual_data_path('BAUM1S_AU')
    capture_visual_au_npys(input_path=BAUM1S_DATA_PATH, output_path=output_path)

    output_path = get_visual_data_path('BAUM1S_RGB')
    capture_visual_npys1(input_path=BAUM1S_DATA_PATH, output_path=output_path, class_labels=BAUM1S_LABELS)

    #output_path = get_visual_data_path('BAUM1S')
    #capture_visual_npys(input_path=BAUM1S_DATA_PATH, output_path=output_path, class_labels=BAUM1S_LABELS)
    
    #output_path = get_visual_data_path('BAUM1S_HISTOGRAM_FRAMES_COUNT')
    #capture_visual_npys(input_path=BAUM1S_DATA_PATH, output_path=output_path, class_labels=BAUM1S_LABELS, selector='histogram_frames_count')
    
    output_path = get_visual_data_path('BAUM1S_FSIM_FRAMES_COUNT')
    capture_visual_npys(input_path=BAUM1S_DATA_PATH, output_path=output_path, class_labels=BAUM1S_LABELS, selector='fsim_frames_count')
    
    #output_path = get_visual_data_path('BAUM1S_SSIM_FRAMES_COUNT')
    #capture_visual_npys(input_path=BAUM1S_DATA_PATH, output_path=output_path, class_labels=BAUM1S_LABELS, selector='ssim_frames_count')
    
    #output_path = get_visual_data_path('BAUM1S_FSM_FRAMES_COUNT')
    #capture_visual_npys(input_path=BAUM1S_DATA_PATH, output_path=output_path, class_labels=BAUM1S_LABELS, selector='fsm_frames_count')

    #output_path = get_visual_data_path('BAUM1S_HISTOGRAM')
    #capture_visual_npys(input_path=BAUM1S_DATA_PATH, output_path=output_path, class_labels=BAUM1S_LABELS, selector='histogram')
    
    #output_path = get_visual_data_path('BAUM1S_FSIM')
    #capture_visual_npys(input_path=BAUM1S_DATA_PATH, output_path=output_path, class_labels=BAUM1S_LABELS, selector='fsim')
    
    #output_path = get_visual_data_path('BAUM1S_SSIM')
    #capture_visual_npys(input_path=BAUM1S_DATA_PATH, output_path=output_path, class_labels=BAUM1S_LABELS, selector='ssim')
    
    #output_path = get_visual_data_path('BAUM1S_FSM')
    #capture_visual_npys(input_path=BAUM1S_DATA_PATH, output_path=output_path, class_labels=BAUM1S_LABELS, selector='fsm')