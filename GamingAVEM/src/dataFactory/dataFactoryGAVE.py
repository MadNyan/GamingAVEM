import sys
sys.path.append('./')

from src.dataPreprocess.dataClassify import *
from src.dataPreprocess.preprocessVisual import *
from src.dataPreprocess.preprocessAudio import *
from src.common.utilities import * 

if __name__ == '__main__':

    classify_GAVE()
    
    output_path = get_audio_data_path('GAVE')
    capture_audio_files(input_path=GAVE_DATA_PATH, output_path=output_path, class_labels=GAVE_LABELS, sampling=16000)

    output_path = get_visual_data_path('GAVE_AU')
    capture_visual_au_npys(input_path=GAVE_DATA_PATH, output_path=output_path)

    output_path = get_visual_data_path('GAVE_RGB')
    capture_visual_npys1(input_path=GAVE_DATA_PATH, output_path=output_path, class_labels=GAVE_LABELS)

    #output_path = get_visual_data_path('GAVE')
    #capture_visual_npys(input_path=GAVE_DATA_PATH, output_path=output_path, class_labels=GAVE_LABELS)

    #output_path = get_visual_data_path('GAVE_HISTOGRAM_FRAMES_COUNT')
    #capture_visual_npys(input_path=GAVE_DATA_PATH, output_path=output_path, class_labels=GAVE_LABELS, selector='histogram_frames_count')
    
    output_path = get_visual_data_path('GAVE_FSIM_FRAMES_COUNT')
    capture_visual_npys(input_path=GAVE_DATA_PATH, output_path=output_path, class_labels=GAVE_LABELS, selector='fsim_frames_count')
    
    #output_path = get_visual_data_path('GAVE_SSIM_FRAMES_COUNT')
    #capture_visual_npys(input_path=GAVE_DATA_PATH, output_path=output_path, class_labels=GAVE_LABELS, selector='ssim_frames_count')
    
    #output_path = get_visual_data_path('GAVE_FSM_FRAMES_COUNT')
    #capture_visual_npys(input_path=GAVE_DATA_PATH, output_path=output_path, class_labels=GAVE_LABELS, selector='fsm_frames_count')

    #output_path = get_visual_data_path('GAVE_HISTOGRAM')
    #capture_visual_npys(input_path=GAVE_DATA_PATH, output_path=output_path, class_labels=GAVE_LABELS, selector='histogram')
    
    #output_path = get_visual_data_path('GAVE_FSIM')
    #capture_visual_npys(input_path=GAVE_DATA_PATH, output_path=output_path, class_labels=GAVE_LABELS, selector='fsim')
    
    #output_path = get_visual_data_path('GAVE_SSIM')
    #capture_visual_npys(input_path=GAVE_DATA_PATH, output_path=output_path, class_labels=GAVE_LABELS, selector='ssim')
    
    #output_path = get_visual_data_path('GAVE_FSM')
    #capture_visual_npys(input_path=GAVE_DATA_PATH, output_path=output_path, class_labels=GAVE_LABELS, selector='fsm')