import cv2
import os, time
import numpy as np
import matplotlib.pyplot as plt


def fun_Hash(img): # 输入为RGB三通道图像
    # 感知哈希算法
    # 缩放32*32
    img = cv2.resize(img, (32, 32))   # , interpolation=cv2.INTER_CUBIC

    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 将灰度图转为浮点型，再进行dct变换
    dct = cv2.dct(np.float32(gray))
    # opencv实现的掩码操作
    dct_roi = dct[0:8, 0:8]

    hash_ = []
    avreage = np.mean(dct_roi)
    for i in range(dct_roi.shape[0]):
        for j in range(dct_roi.shape[1]):
            if dct_roi[i, j] > avreage:
                hash_.append(1)
            else:
                hash_.append(0)
    # 输出为hash列表
    return hash_

def com2hashstr(hash1, hash2): # 计算两个哈希值的相似度
    if len(hash1) != len(hash2):
        return -1
    n = 0
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            n = n + 1
    n = sorce_remap(n, 0, 64, -1)
    return n

def sorce_remap(value, limit1, limit2, sign): # 将相似度映射到0，1之间
    N_long = 100/(limit2 - limit1)
    if sign == 1:
        #return float(value*N_long/100)
        return value*N_long
    elif sign == -1:
        #return float((100 - value*N_long)/100)
        return 100 - value*N_long

def segvideo_10(vcpath, save_root): # 将视频以fps=10拆解图像，并保存
    videoCapture = cv2.VideoCapture(vcpath)
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    cnt = 0
    cnt_name = 0
    while videoCapture.isOpened():
        ret, frame = videoCapture.read()
        if not ret:
            break
        if cnt % int(fps/10) == 0:
            img_savepath = os.path.join(save_root, str(cnt_name) + '.jpg')
            cv2.imwrite(img_savepath, frame)
            cnt_name = cnt_name + 1
        cnt = cnt + 1        
    return fps

def get_imgs_mean_value(imgs_root): # 获得三通道图像目录的像素变化函数
    cnt = 0
    mean_list = []
    for n in range(len(os.listdir(imgs_root))):
        img_path = os.path.join(imgs_root, str(n) + '.jpg')
        img = cv2.imread(img_path,0)
        mean_list.append(img.mean())
    #plt.plot(r_list)
    #plt.show()
    #exit()
    mean_list = np.array(mean_list)
    return mean_list

def get_imgs_similar_value(imgs_root): # 获得图像目录相邻图像相似度变化函数
    similar_value_list = []
    cnt_name = 0
    for n in range(len(os.listdir(imgs_root)) - 1):
        img1_path = os.path.join(imgs_root, str(n) + '.jpg')
        img2_path = os.path.join(imgs_root, str(n+1) + '.jpg')
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        hash1 = fun_Hash(img1)
        hash2 = fun_Hash(img2)
        value_similar = com2hashstr(hash1, hash2)
        similar_value_list.append(value_similar)
        cnt_name = cnt_name + 1
    #plt.plot(similar_value_list)
    #plt.show()
    #exit()
    similar_value_list = np.array(similar_value_list)
    return similar_value_list

def cau_score(imgsroot1, imgsroot2, list_index1, list_index2):
    list_imgs1 = os.listdir(imgsroot1)
    list_imgs2 = os.listdir(imgsroot2)
    cnt = 0
    cnt_yes = 0
    for i in range(list_index1[1]-list_index1[0]):
        img1path = os.path.join(imgsroot1, str(i+list_index1[0]) + '.jpg')
        img2path = os.path.join(imgsroot2, str(i+list_index2[0]) + '.jpg')
        temp_img1 = cv2.imread(img1path)
        temp_img2 = cv2.imread(img2path)
        hash1 = fun_Hash(temp_img1)
        hash2 = fun_Hash(temp_img2)
        temp_score = com2hashstr(hash1, hash2)
        if temp_score > 80:
            cnt_yes = cnt_yes + 1
        cnt = cnt + 1
    return cnt_yes/cnt

def main(imgsroot1, imgsroot2):
    if len(os.listdir(imgsroot2)) > len(os.listdir(imgsroot1)):
        temp = imgsroot1
        imgsroot1 = imgsroot2
        imgsroot2 = temp
    limit1 = 1
    list_dst1 = get_imgs_mean_value(imgsroot1)
    list_dst2 = get_imgs_mean_value(imgsroot2)
    if np.var(list_dst1)<limit1 and np.var(list_dst1)<limit1:
        list_dst1 = get_imgs_similar_value(imgsroot1)
        list_dst2 = get_imgs_similar_value(imgsroot2)
        if np.var(list_dst1)<limit1 and np.var(list_dst1)<limit1:
            list_dst1 = []
            list_dst2 = []
    if len(list_dst1)==0 and len(list_dst2)==0:
        score = cau_score(imgsroot1, imgsroot2, [0,30], [0,30])
        if score > 0.8:
            return True
        else:
            return False
    # 找到最相似的五个片段
    similar_list1, similar_list2 = find_similar_sent(list_dst1, list_dst2, compare_long = 50, limit = 1)
    print(similar_list1, similar_list2)
    for n,m in zip(similar_list1,similar_list2):
        score_2 = cau_score(imgsroot1, imgsroot2, n, m)
        if score_2 > 0.8:
            return True
    return False

def find_similar_sent(list_1, list_2, compare_long = 50, limit = 1): # 根据像素比变化函数，找到列表相似片段
    if len(list_1) < compare_long or len(list_2) < compare_long:
        return None
    #list_1 = np.array(list_1)
    #list_2 = np.array(list_2)
    if len(list_2) > len(list_1):
        list_long = list_2
        list_short = list_1
    else:
        list_long = list_1
        list_short = list_2
    similar_list_long = []
    similar_list_short = []
    i = 0
    while True:
        if len(similar_list_long) > 4:
            break
        if not i < (len(list_long) - compare_long):
            break
        temp_list_long = list_long[i:i+compare_long]
        j = 0
        sign_tt = 0
        while True:
            if not j < (len(list_short) - compare_long):
                break
            temp_list_short = list_short[j:j+compare_long]
            
            diff_list = temp_list_long - temp_list_short
            var = np.var(diff_list)
            if var < limit:
                similar_list_long.append([i,i+compare_long])
                similar_list_short.append([j,j+compare_long])
                sign_tt = 1
                j = j + compare_long
            else:
                j = j + 1
        if sign_tt == 1:
            i = i + compare_long
            sign_tt = 0
        else:
            i = i + 1
    return similar_list_long, similar_list_short
    
if __name__ == "__main__":
    t1 = time.time()
    root_1 = 'imgs1'
    root_2 = 'imgs2'
    vcpath1 = '1-2.mp4'
    vcpath2 = '2-2.mp4'
    list_1 = os.listdir(root_1)
    for n in list_1:
        path = os.path.join(root_1, n)
        os.remove(path)
    list_2 = os.listdir(root_2)
    for n in list_2:
        path = os.path.join(root_2, n)
        os.remove(path)
    t2 = time.time()
    segvideo_10(vcpath1, root_1)
    segvideo_10(vcpath2, root_2)
    t3 = time.time()
    sign = main(root_1, root_2)
    print('两视频是否含有相似片段: ', sign)
    print('time long: ',len(os.listdir(root_1))/10, len(os.listdir(root_2))/10)
    print('init imgs cost: ', t2-t1)
    print('seg video cost: ', t3-t2)
    print('fun cost: ', time.time() - t3)
