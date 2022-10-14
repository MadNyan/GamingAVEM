import sys
sys.path.append('./')

from src.dataPreprocess.preprocessVisual import *
from src.common.utilities import * 

if __name__ == '__main__':

    output_path = get_visual_data_path('SAVEE_AU')
    capture_visual_au_npys(input_path=SAVEE_DATA_PATH, output_path=output_path)

    output_path = get_visual_data_path('SAVEE_RGB')
    capture_visual_npys1(input_path=SAVEE_DATA_PATH, output_path=output_path, class_labels=SAVEE_LABELS)

    #output_path = get_visual_data_path('SAVEE')
    #capture_visual_npys(input_path=SAVEE_DATA_PATH, output_path=output_path, class_labels=SAVEE_LABELS, time_F=2)

    #output_path = get_visual_data_path('SAVEE_HISTOGRAM_FRAMES_COUNT')
    #capture_visual_npys(input_path=SAVEE_DATA_PATH, output_path=output_path, class_labels=SAVEE_LABELS, time_F=2, selector='histogram_frames_count')
    
    output_path = get_visual_data_path('SAVEE_FSIM_FRAMES_COUNT')
    capture_visual_npys(input_path=SAVEE_DATA_PATH, output_path=output_path, class_labels=SAVEE_LABELS, time_F=2, selector='fsim_frames_count')
    
    #output_path = get_visual_data_path('SAVEE_SSIM_FRAMES_COUNT')
    #capture_visual_npys(input_path=SAVEE_DATA_PATH, output_path=output_path, class_labels=SAVEE_LABELS, time_F=2, selector='ssim_frames_count')
    
    #output_path = get_visual_data_path('SAVEE_FSM_FRAMES_COUNT')
    #capture_visual_npys(input_path=SAVEE_DATA_PATH, output_path=output_path, class_labels=SAVEE_LABELS, time_F=2, selector='fsm_frames_count')
    
    #output_path = get_visual_data_path('SAVEE_HISTOGRAM')
    #capture_visual_npys(input_path=SAVEE_DATA_PATH, output_path=output_path, class_labels=SAVEE_LABELS, time_F=2, selector='histogram')
    
    #output_path = get_visual_data_path('SAVEE_FSIM')
    #capture_visual_npys(input_path=SAVEE_DATA_PATH, output_path=output_path, class_labels=SAVEE_LABELS, time_F=2, selector='fsim')
    
    #output_path = get_visual_data_path('SAVEE_SSIM')
    #capture_visual_npys(input_path=SAVEE_DATA_PATH, output_path=output_path, class_labels=SAVEE_LABELS, time_F=2, selector='ssim')
    
    #output_path = get_visual_data_path('SAVEE_FSM')
    #capture_visual_npys(input_path=SAVEE_DATA_PATH, output_path=output_path, class_labels=SAVEE_LABELS, time_F=2, selector='fsm')