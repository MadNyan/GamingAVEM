from array import array
import os
import cv2
import numpy as np
import pandas as pd
from src.common.utilities import *
from src.common.qualityMetrics import *

def get_frames_from_video(video_name, time_F=1):
    video_frames = []
    vc = cv2.VideoCapture(video_name)
    c = 1
    if vc.isOpened(): #判斷是否開啟影片
        rval, video_frame = vc.read()
    else:
        rval = False
    while rval:   #擷取視頻至結束
        if c % time_F == 0: #每隔幾幀進行擷取
            video_frames.append(video_frame)     
        c = c + 1
        rval, video_frame = vc.read()
    vc.release()
    return video_frames

def crop_image_face(image, resize_x, resize_y):
    #是否已裁切
    is_crop = False
    #取得分類器
    face_cascade = cv2.CascadeClassifier('./classifier/haarcascade_frontalface_alt2.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #偵測臉部
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=5, minSize=(32, 32))
    if len(faces) > 0:
        x = faces[0][0]
        y = faces[0][1]
        w = faces[0][2]
        h = faces[0][3]
        image = image[y : y + h, x : x + w]
        m, n, _ = image.shape
        if m < resize_x or n < resize_y:
            i = cv2.INTER_CUBIC
        else:
            i = cv2.INTER_AREA
        image = cv2.resize(image, (resize_x, resize_y), interpolation=i)
        is_crop = True
    return is_crop, image

def crop_image_face_with_caffemodel(image, resize_x, resize_y):
    #是否已裁切
    is_crop = False
    (h, w) = image.shape[:2]
    #取得分類器
    PROTOTXT = './classifier/deploy.prototxt.txt'
    MODEL = './classifier/res10_300x300_ssd_iter_140000.caffemodel'
    net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    #偵測臉部
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), scalefactor=0.4, size=(300, 300), mean=(104, 117, 123))
    net.setInput(blob)
    preds = net.forward()
    face = None

    for k in range(0, preds.shape[2]):
        confidence = preds[0, 0, k, 2]
        if confidence > 0.5:
            box = preds[0, 0, k, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype('int')
            face = image[startY:endY, startX:endX]

            m, n, _ = face.shape
            if m < resize_x or n < resize_y:
                i = cv2.INTER_CUBIC
            else:
                i = cv2.INTER_AREA

            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (resize_x, resize_x), interpolation=i)
            is_crop = True
            break

    return is_crop, face

def get_image_histogram(image):
    # 創建 RGB 三通道長條圖（長條圖矩陣）
    h, w, c = image.shape
    # 創建一個（16*16*16,1）的初始矩陣，作為長條圖矩陣
    # 16*16*16的意思為三通道每通道有16個bins
    rgbhist = np.zeros([16 * 16 * 16, 1], np.float32)
    bsize = 256 / 16
    for row in range(h):
        for col in range(w):
            b = image[row, col, 0]
            g = image[row, col, 1]
            r = image[row, col, 2]
            # 人為構建長條圖矩陣的索引，該索引是通過每一個圖元點的三通道值進行構建
            index = int(b / bsize) * 16 * 16 + int(g / bsize) * 16 + int(r / bsize)
           	# 該處形成的矩陣即為長條圖矩陣
            rgbhist[int(index), 0] += 1
    return rgbhist

def hist_compare(image1, image2):
    # 長條圖比較函數
    # 創建第一幅圖的rgb三通道長條圖（長條圖矩陣）
    hist1 = get_image_histogram(image1)
    # 創建第二幅圖的rgb三通道長條圖（長條圖矩陣）
    hist2 = get_image_histogram(image2)
    # 進行三種方式的長條圖比較
    match = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
    #print(str(match) + '\n')
    return match

def key_frames_selector(selector='', frames=[], frames_range=3, strides=4, frames_count=10, is_reverse=False):
    key_frames = []
    indexes = []
    if selector == 'histogram':
        key_frames, indexes = key_frames_selector_with_histogram(frames, frames_range, strides, is_reverse)
    elif selector == 'fsim':
        key_frames, indexes = key_frames_selector_with_fsim(frames, frames_range, strides, is_reverse)
    elif selector == 'ssim':
        key_frames, indexes = key_frames_selector_with_ssim(frames, frames_range, strides, is_reverse)
    elif selector == 'fsm':
        key_frames, indexes = key_frames_selector_with_fsm(frames, frames_range, strides, is_reverse)
    elif selector == 'histogram_frames_count':
        key_frames, indexes = key_frames_selector_with_histogram_frames_count(frames, frames_count, is_reverse)
    elif selector == 'fsim_frames_count':
        key_frames, indexes = key_frames_selector_with_fsim_frames_count(frames, frames_count, is_reverse)
    elif selector == 'ssim_frames_count':
        key_frames, indexes = key_frames_selector_with_ssim_frames_count(frames, frames_count, is_reverse)
    elif selector == 'fsm_frames_count':
        key_frames, indexes = key_frames_selector_with_fsm_frames_count(frames, frames_count, is_reverse)
    else:
        key_frames = frames
        for index in range(len(frames)):
            indexes.append(index)
    return key_frames, indexes

def key_frames_selector_with_histogram(frames, frames_range=3, strides=4, is_reverse=False):
    frames_range = frames_range * 2 + 1
    hists = []
    key_frames = []
    indexes = []
    # 計算每相鄰影格的長條圖差異
    for i in range(0, len(frames)):
        hists.append(get_image_histogram(frames[i]))
    j = 0
    last_index = -1
    while j + frames_range <= len(hists):
        # 是否初始化
        is_init = False
        # 比較次數會減一
        for k in range(j, j + frames_range - 1):
            # 避免取得重複資料
            if last_index == k:
                continue
            difference = cv2.compareHist(hists[k], hists[k+1], cv2.HISTCMP_CHISQR)
            # 初始化與紀錄最小值
            if is_init == False or (is_reverse == False and min_difference > difference) or (is_reverse == True and min_difference < difference):
                min_difference = difference
                index = k
                is_init = True
        key_frames.append(frames[index])
        indexes.append(index)
        last_index = index
        print('selected no.' + str(index) + ' frame to key frame')
        j = j + strides
    return key_frames, indexes

def key_frames_selector_with_fsim(frames, frames_range=3, strides=4, is_reverse=False):
    frames_range = frames_range * 2 + 1
    key_frames = []
    indexes = []
    j = 0
    last_index = -1
    while j + frames_range <= len(frames):
        # 是否初始化
        is_init = False
        # 比較次數會減一
        for k in range(j, j + frames_range - 1):
            # 避免取得重複資料
            if last_index == k:
                continue
            m = fsim(frames[k], frames[k+1])
            # 初始化與紀錄最小值
            if is_init == False or (is_reverse == False and _max < m) or (is_reverse == True and _max > m):
                _max = m
                index = k
                is_init = True
        key_frames.append(frames[index])
        indexes.append(index)
        last_index = index
        print('selected no.' + str(index) + ' frame to key frame')
        j = j + strides
    return key_frames, indexes

def key_frames_selector_with_ssim(frames, frames_range=3, strides=4, is_reverse=False):
    frames_range = frames_range * 2 + 1
    key_frames = []
    indexes = []
    j = 0
    last_index = -1
    while j + frames_range <= len(frames):
        # 是否初始化
        is_init = False
        # 比較次數會減一
        for k in range(j, j + frames_range - 1):
            # 避免取得重複資料
            if last_index == k:
                continue
            m = ssim(frames[k], frames[k+1])
            # 初始化與紀錄最小值
            if is_init == False or (is_reverse == False and _max < m) or (is_reverse == True and _max > m):
                _max = m
                index = k
                is_init = True
        key_frames.append(frames[index])
        indexes.append(index)
        last_index = index
        print('selected no.' + str(index) + ' frame to key frame')
        j = j + strides
    return key_frames, indexes

def key_frames_selector_with_fsm(frames, frames_range=3, strides=4, is_reverse=False):
    frames_range = frames_range * 2 + 1
    key_frames = []
    indexes = []
    j = 0
    last_index = -1
    while j + frames_range <= len(frames):
        # 是否初始化
        is_init = False
        # 比較次數會減一
        for k in range(j, j + frames_range - 1):
            # 避免取得重複資料
            if last_index == k:
                continue
            m = fsm(frames[k], frames[k+1])
            # 初始化與紀錄最小值
            if is_init == False or (is_reverse == False and _max < m) or (is_reverse == True and _max > m):
                _max = m
                index = k
                is_init = True
        key_frames.append(frames[index])
        indexes.append(index)
        last_index = index
        print('selected no.' + str(index) + ' frame to key frame')
        j = j + strides
    return key_frames, indexes

def key_frames_selector_with_histogram_frames_count(frames, frames_count=10, is_reverse=False):
    strides = int(len(frames) / (frames_count + 1))
    frames_range = strides * 2
    hists = []
    key_frames = []
    indexes = []
    # 計算每相鄰影格的長條圖差異
    for i in range(0, len(frames)):
        hists.append(get_image_histogram(frames[i]))
    j = 0
    last_index = -1
    times = 0
    while j + frames_range <= len(hists) and times < frames_count:
        # 是否初始化
        is_init = False
        # 比較次數會減一
        for k in range(j, j + frames_range - 1):
            # 避免取得重複資料
            if last_index == k:
                continue
            difference = cv2.compareHist(hists[k], hists[k+1], cv2.HISTCMP_CHISQR)
            # 初始化與紀錄最小值
            if is_init == False or (is_reverse == False and min_difference > difference) or (is_reverse == True and min_difference < difference):
                min_difference = difference
                index = k
                is_init = True
        key_frames.append(frames[index])
        indexes.append(index)
        last_index = index
        times = times + 1
        print('selected no.' + str(index) + ' frame to key frame')
        j = j + strides
    return key_frames, indexes

def key_frames_selector_with_fsim_frames_count(frames, frames_count=10, is_reverse=False):
    strides = int(len(frames) / (frames_count + 1))
    frames_range = strides * 2
    key_frames = []
    indexes = []
    j = 0
    last_index = -1
    times = 0
    while j + frames_range <= len(frames) and times < frames_count:
        # 是否初始化
        is_init = False
        # 比較次數會減一
        for k in range(j, j + frames_range - 1):
            # 避免取得重複資料
            if last_index == k:
                continue
            m = fsim(frames[k], frames[k+1])
            # 初始化與紀錄最小值
            if is_init == False or (is_reverse == False and _max < m) or (is_reverse == True and _max > m):
                _max = m
                index = k
                is_init = True
        key_frames.append(frames[index])
        indexes.append(index)
        last_index = index
        times = times + 1
        print('selected no.' + str(index) + ' frame to key frame')
        j = j + strides
    return key_frames, indexes

def key_frames_selector_with_ssim_frames_count(frames, frames_count=10, is_reverse=False):
    strides = int(len(frames) / (frames_count + 1))
    frames_range = strides * 2
    key_frames = []
    indexes = []
    j = 0
    last_index = -1
    times = 0
    while j + frames_range <= len(frames) and times < frames_count:
        # 是否初始化
        is_init = False
        # 比較次數會減一
        for k in range(j, j + frames_range - 1):
            # 避免取得重複資料
            if last_index == k:
                continue
            m = ssim(frames[k], frames[k+1])
            # 初始化與紀錄最小值
            if is_init == False or (is_reverse == False and _max < m) or (is_reverse == True and _max > m):
                _max = m
                index = k
                is_init = True
        key_frames.append(frames[index])
        indexes.append(index)
        last_index = index
        times = times + 1
        print('selected no.' + str(index) + ' frame to key frame')
        j = j + strides
    return key_frames, indexes

def key_frames_selector_with_fsm_frames_count(frames, frames_count=10, is_reverse=False):
    strides = int(len(frames) / (frames_count + 1))
    frames_range = strides * 2
    key_frames = []
    indexes = []
    j = 0
    last_index = -1
    times = 0
    while j + frames_range <= len(frames) and times < frames_count:
        # 是否初始化
        is_init = False
        # 比較次數會減一
        for k in range(j, j + frames_range - 1):
            # 避免取得重複資料
            if last_index == k:
                continue
            m = fsm(frames[k], frames[k+1])
            # 初始化與紀錄最小值
            if is_init == False or (is_reverse == False and _max < m) or (is_reverse == True and _max > m):
                _max = m
                index = k
                is_init = True
        key_frames.append(frames[index])
        indexes.append(index)
        last_index = index
        times = times + 1
        print('selected no.' + str(index) + ' frame to key frame')
        j = j + strides
    return key_frames, indexes

def capture_visual_npys(input_path=ENTERFACE_DATA_PATH, output_path=ENTERFACE_VISUAL_NPY_DATA_PATH, class_labels=ENTERFACE_LABELS, time_F=1,
                        resize_x=224, resize_y=224, to_local_binary_pattern=False, to_interlaced_derivative_pattern=False, selector='',is_reverse=False , key_frames_count=0):
    #time_F越小，取樣張數越多
    for lab in os.listdir(input_path):
        _lab = lab + '/'
        for file_name in os.listdir(input_path + _lab):

            if 'enterface' in input_path and 's6' in file_name:
                continue;
        
            name = file_name.split('.')[0]
            _input_path = input_path + _lab + file_name
            _output_path = output_path + _lab

            make_dirs(_output_path)

            if os.path.isfile(_output_path + name + '.npy') == True:
                print(name + ' is captured')
                continue
            
            print('started capturing ' + name)

            capture_visual_npy(input_path=_input_path, output_path=_output_path, file_name=name, time_F=time_F, resize_x=resize_x, resize_y=resize_y, 
                               to_local_binary_pattern=to_local_binary_pattern, to_interlaced_derivative_pattern=to_interlaced_derivative_pattern, 
                               selector=selector,is_reverse=is_reverse , key_frames_count=key_frames_count)

            print('ended capturing ' + name)            

    bind_visual_npys(input_path=output_path, output_path=output_path, class_labels=class_labels)

def capture_visual_npy(input_path='', output_path='', file_name='default', time_F=1, resize_x=224, resize_y=224, 
                       to_local_binary_pattern=False, to_interlaced_derivative_pattern=False, selector='',is_reverse=False , key_frames_count=0):
    
    video_frames = get_frames_from_video(input_path, time_F) #讀取影片並轉成圖片

    _video_frames = []
    for i in range(0, len(video_frames)): #將影格裁切
        is_crop, image = crop_image_face_with_caffemodel(video_frames[i], resize_x, resize_y)
        if is_crop:
            _video_frames.append(image)

    key_frames, indexes = key_frames_selector(selector=selector, frames=_video_frames, is_reverse=is_reverse) #影格選擇

    last_img = []
    imgs = []
    new_video_frames = []
    csv = ''
    key_frames_range = key_frames_count
    if key_frames_count == 0:
        key_frames_range = len(key_frames)
    for i in range(0, key_frames_range): #輸出關鍵影格
        if i < len(key_frames):
            last_img = []
            gray_scale = get_mean_gray_scale(key_frames[i])
            last_img.append(gray_scale)
            if to_local_binary_pattern:
                lbp_image = get_local_binary_pattern(key_frames[i])
                last_img.append(lbp_image)
            if to_interlaced_derivative_pattern:
                idp_image = get_interlaced_derivative_pattern(key_frames[i])     
                last_img.append(idp_image)           
            print('added no.' + str(indexes[i]) + ' frame')
            csv = csv + str(indexes[i]) + ','

            rgb_output_path = output_path + file_name + '_rgb/'
            make_dirs(rgb_output_path)
            cv2.imwrite(rgb_output_path + 'output' + str(indexes[i]) + '.png', key_frames[i])

            rgb_output_path = output_path + file_name + '_video/'
            make_dirs(rgb_output_path)
            cv2.imwrite(rgb_output_path + 'output' + str(indexes[i]) + '.png', video_frames[indexes[i]])
            new_video_frames.append(video_frames[indexes[i]])

            gray_output_path = output_path + file_name + '_gray/'
            make_dirs(gray_output_path)
            cv2.imwrite(gray_output_path + 'output' + str(indexes[i]) + '_gray.png', gray_scale)
        else:
            gray_scale = last_img[0]
            if to_local_binary_pattern:
                lbp_image = last_img[1]
            if to_interlaced_derivative_pattern:
                idp_image = last_img[2]
            print('added the previous frame')
            csv = csv + 'previous,'
        px_x = []
        for x in range(0, resize_x):
            px_y = []
            for y in range(0, resize_y):
                px = []
                px.append(gray_scale[x][y])
                if to_local_binary_pattern:
                    px.append(lbp_image[x][y])                    
                if to_interlaced_derivative_pattern:
                    px.append(idp_image[x][y])
                px_y.append(px)
            px_x.append(px_y)
        imgs.append(px_x)
    if len(imgs) > 0:
        write_result(output_path + file_name + '.csv', csv)
        np.save(output_path + file_name + '.npy', np.array(imgs))
        create_video(output_path=output_path + file_name + '.mp4', frames=new_video_frames)
    print(np.array(imgs).shape)

def bind_visual_npys(input_path=ENTERFACE_VISUAL_NPY_DATA_PATH, output_path=ENTERFACE_VISUAL_NPY_DATA_PATH, class_labels=ENTERFACE_LABELS):
    
    file_name = get_labels_name(class_labels)
    x_path = output_path + file_name + 'x.npy'
    y_path = output_path + file_name + 'y.npy'
    if os.path.isfile(x_path) == True and os.path.isfile(y_path) == True:
        print(x_path + ' is captured')
        print(y_path + ' is captured')
        return 0
    
    remove_file(x_path)
    remove_file(y_path)

    data = []
    labels = []
    names = []
    for i, directory in enumerate(class_labels):
        folder = input_path + directory + '/'
        print('started reading folder %s' % directory)
        for filename in os.listdir(folder):
            if '.npy' in filename:
                filepath = folder + filename
                _data = np.load(filepath, allow_pickle=True)
                data.append(_data)
                labels.append(i)
                names.append(filename)
        print('ended reading folder %s' % directory)
    np.save(x_path, data)
    np.save(y_path, labels)
    print('ended capturing')

def create_video(output_path='', frames=[]):
    if len(frames) > 0:
        # 使用 OpenCV 將圖像轉換成影片
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MP4V'), 30, (frames[0].shape[1], frames[0].shape[0]))
        for frame in frames:
            out.write(frame)
        out.release()

def get_mean_gray_scale(image):
    m, n, _ = image.shape
    gray_scale = np.zeros((m, n),np.uint8) 
    # converting image to grayscale
    for i in range(0,m):
        for j in range(0,n):
            b = image[i, j, 0]
            g = image[i, j, 1]
            r = image[i, j, 2]
            gray = (int(b) + int(g) + int(r)) / 3
            gray_scale[i, j] = gray
 
    return gray_scale

def get_local_binary_pattern(image, is_gray_scale=False):
    def assign_bit(picture, x, y, c):   #comparing bit with threshold value of centre pixel
        bit = 0  
        try:
            if picture[x][y] >= c:
                bit = 1
        except:
            pass
        return bit
    
    def local_bin_val(picture, x, y):  #calculating local binary pattern value of a pixel
        eight_bit_binary = []
        centre = picture[x][y] 
        powers = [1, 2, 4, 8, 16, 32, 64, 128] 
        decimal_val = 0
        #starting from frames_count right,assigning bit to pixels clockwise 
        eight_bit_binary.append(assign_bit(picture, x-1, y + 1,centre)) 
        eight_bit_binary.append(assign_bit(picture, x, y + 1, centre)) 
        eight_bit_binary.append(assign_bit(picture, x + 1, y + 1, centre)) 
        eight_bit_binary.append(assign_bit(picture, x + 1, y, centre)) 
        eight_bit_binary.append(assign_bit(picture, x + 1, y-1, centre)) 
        eight_bit_binary.append(assign_bit(picture, x, y-1, centre)) 
        eight_bit_binary.append(assign_bit(picture, x-1, y-1, centre)) 
        eight_bit_binary.append(assign_bit(picture, x-1, y, centre))     
        #calculating decimal value of the 8-bit binary number
        for i in range(len(eight_bit_binary)): 
            decimal_val += eight_bit_binary[i] * powers[i] 
            
        return decimal_val 
    
    m, n, _ = image.shape
    gray_scale = image
    if is_gray_scale == False:
        gray_scale = get_mean_gray_scale(image)  #converting image to grayscale
    lbp_image = np.zeros((m, n),np.uint8) 
    # converting image to lbp
    for i in range(0,m): 
        for j in range(0,n): 
            lbp_image[i, j] = local_bin_val(gray_scale, i, j) 
 
    return lbp_image

def get_interlaced_derivative_pattern(image, is_gray_scale=False):
    def assign_bit(picture, x, y, c1, c2, d):  #assign bit according to degree and neighbouring pixel
        a = 0    
        b = 0 
        #a and b are 1 if increasing and 0 if decreasing
        if (d == 0):
            try:
                a = int(picture[c1][c2]) - int(picture[c1+1][c2])
                b = int(picture[x][y]) - int(picture[x+1][y])
            except:
                pass
        if (d == 45):
            try:
                a = int(picture[c1][c2]) - int(picture[c1+1][c2+1])
                b = int(picture[x][y]) - int(picture[x+1][y+1])
            except:
                pass
        if (d == 90):
            try:
                a = int(picture[c1][c2]) - int(picture[c1][c2+1])
                b = int(picture[x][y]) - int(picture[x][y+1])
            except:
                pass
        if (d == 135):
            try:
                a = int(picture[c1][c2]) - int(picture[c1-1][c2+1])
                b = int(picture[x][y]) - int(picture[x-1][y+1])
            except:
                pass
        if (a >= b): #if monotonically increasing or decreasing than 0
            return 1
        else:        #if turning point 
            return 0

    def local_bin_val(picture, x, y):  #calculating interlaced derivative pattern value of a pixel
        eight_bit_binary = []
        c1 = x
        c2 = y
        powers = [1, 2, 4, 8, 16, 32, 64, 128] 
        decimal_val = 0
        #starting from frames_count right,assigning bit to pixels clockwise 
        eight_bit_binary.append(assign_bit(picture, x - 1, y + 1, c1, c2, 135))
        eight_bit_binary.append(assign_bit(picture, x, y + 1, c1, c2, 90))
        eight_bit_binary.append(assign_bit(picture, x + 1, y + 1, c1, c2, 45))
        eight_bit_binary.append(assign_bit(picture, x + 1, y, c1, c2, 0))
        eight_bit_binary.append(assign_bit(picture, x + 1, y - 1, c1, c2, 135))
        eight_bit_binary.append(assign_bit(picture, x, y - 1, c1, c2, 90))
        eight_bit_binary.append(assign_bit(picture, x - 1, y - 1, c1, c2, 45))
        eight_bit_binary.append(assign_bit(picture, x - 1, y, c1, c2, 0))
        #calculating decimal value of the 8-bit binary number
        for i in range(len(eight_bit_binary)): 
            decimal_val += eight_bit_binary[i] * powers[i] 
            
        return decimal_val 
    
    m, n, _ = image.shape 
    gray_scale = image
    if is_gray_scale == False:
        gray_scale = get_mean_gray_scale(image)  #converting image to grayscale
    idp_image = np.zeros((m, n),np.uint8) 
    # converting image to idp
    for i in range(0,m): 
        for j in range(0,n): 
            idp_image[i, j] = local_bin_val(gray_scale, i, j) 
 
    return idp_image

def capture_visual_au_npys(input_path=ENTERFACE_DATA_PATH, output_path=ENTERFACE_VISUAL_AU_DATA_PATH):
    #time_F越小，取樣張數越多
    for lab in os.listdir(input_path):
        _lab = lab + '/'
        for file_name in os.listdir(input_path + _lab):

            if 'enterface' in input_path and 's6' in file_name:
                continue;

            if '.mp4' not in file_name:
                continue;
        
            name = file_name.split('.')[0]
            _input_path = input_path + _lab
            _output_path = output_path + _lab

            make_dirs(_output_path)

            if os.path.isfile(_output_path + name + '.npy') == True:
                print(name + ' is captured')
                continue
            
            print('started capturing ' + name)

            if extract_features_OpenFace(file_name=file_name, input_path=_input_path, openFace_path='D:/Adair/GamingAVEM/GamingAVEM/classifier/OpenFace/FeatureExtraction.exe', out_path=_output_path):
                print('ended capturing ' + name)
            else:
                print('capturing ' + name + ' fail')

def extract_features_OpenFace(file_name, input_path, openFace_path, out_path, static=" -au_static"):
    name = file_name.split(".")[0]
    out_path_video = os.path.join(out_path, name)
    os.makedirs(out_path_video, exist_ok=True)
    video_path = os.path.join(input_path, file_name)
    
    # 使用OpenFace擷取特徵
    command = openFace_path + " "+ static + " -f "+video_path+" -out_dir "+out_path_video
    os.system(command)

    path_AU = os.path.join(out_path_video, name+".csv")
    if os.path.isfile(path_AU):
        # 取出AUs特徵
        cols2select = [" AU01_r" ," AU02_r" ," AU04_r" , " AU05_r"," AU06_r", " AU07_r", " AU09_r", " AU10_r"," AU12_r"," AU14_r"," AU15_r", " AU17_r", " AU20_r", " AU23_r", " AU25_r", " AU26_r", " AU45_r", " AU01_c", " AU02_c", " AU04_c" ," AU05_c", " AU06_c", " AU07_c", " AU09_c", " AU10_c", " AU12_c", " AU14_c", " AU15_c", " AU17_c", " AU20_c", " AU23_c", " AU25_c", " AU26_c", " AU28_c", " AU45_c"]
        df_aus = pd.read_csv(path_AU, ",")
        df_aux = df_aus[cols2select]
    
        # 建立csv
        df_aux.to_csv(os.path.join(out_path, name+".csv"), sep=",", header=True, index=False)

        # 建立npy
        data = df_aux.to_numpy()
        np.save(out_path + name + '.npy', data)
        return True
    return False

def capture_visual_rgb_npys(input_path=ENTERFACE_DATA_PATH, output_path=ENTERFACE_VISUAL_NPY_DATA_PATH, class_labels=ENTERFACE_LABELS, time_F=1, resize_x=224, resize_y=224):
    #time_F越小，取樣張數越多
    for lab in os.listdir(input_path):
        _lab = lab + '/'
        for file_name in os.listdir(input_path + _lab):

            if 'enterface' in input_path and 's6' in file_name:
                continue;
        
            name = file_name.split('.')[0]
            _input_path = input_path + _lab + file_name
            _output_path = output_path + _lab

            make_dirs(_output_path)

            if os.path.isfile(_output_path + name + '.npy') == True:
                print(name + ' is captured')
                continue
            
            print('started capturing ' + name)

            
            video_frames = get_frames_from_video(_input_path, time_F) #讀取影片並轉成圖片
            
            _video_frames = []
            for i in range(0, len(video_frames)): #將影格裁切
                is_crop, image = crop_image_face_with_caffemodel(video_frames[i], resize_x, resize_y)
                if is_crop:
                    _video_frames.append(image)
            np.save(_output_path + file_name.split('.')[0] + '.npy', np.array(_video_frames))
            print(np.array(_video_frames).shape)

            print('ended capturing ' + name)