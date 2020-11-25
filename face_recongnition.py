import face_recognition  
"""
截取出图片中的人脸，并单独保存到路径
"""
import glob
import cv2
from PIL import Image
import os
import pickle
import gc
import numpy as np
import tensorflow as tf

class preprocess(object):
    def __init__(self,scale_size=139):
        self.img_cant = []      # 记录没有检测出人脸的图片名称，便于后续处理
        self.scale_size = scale_size    # 检测出人脸后将人脸都resize到scale_size

    # 展示某图片中截取出的人脸
    @staticmethod
    def show_face(path):
        image=face_recognition.load_image_file(path)
        face_locations=face_recognition.face_locations(image)
        if len(face_locations)==0:
            return -1
        for k,i in enumerate(face_locations):
            (a,b,c,d)=i
            image_spilt=image[a:c,d:b,:]
            cv2.imshow('img_{}'.format(k),image_spilt)
            return 1

    def split_face(self,in_path,out_path,use_cnn):
        """
        Parameters:
        in_path: 需要检测人脸的图片的路径，/root/in_path/
        out_path: 检测出的人脸的保存路径，/root/out_path/
        use_cnn: 是否使用cnn网络，False表示使用传统方法，速度快精度较低，True表示使用cnn，速度慢精度较高
        """
        image=face_recognition.load_image_file(in_path)
        # 如果图片太大则等比将长边缩放至2000
        if max(image.shape)>2000:
            if image.shape[0]>image.shape[1]:
                image=cv2.resize(image,(2000,int(2000*image.shape[1]/image.shape[0])))
            else:
                image=cv2.resize(image,(int(2000*image.shape[0]/image.shape[1]),2000))

        if not use_cnn:
            face_locations=face_recognition.face_locations(image)
        else:
            face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=1, model="cnn")
        img_name=os.path.basename(in_path)
        if len(face_locations)==0:
            self.img_cant.append(img_name)
            print(img_name)
            return
        for k,i in enumerate(face_locations):
            (a,b,c,d)=i
            image_spilt=image[a:c,d:b,:]
            image_spilt = self.scale_img(image_spilt)
            img=Image.fromarray(image_spilt)
            img.save(out_path+'/{}_{}.png'.format(img_name,k))
            print('success')

    # 先将短边等比缩放至self.scale_size，再在长边中随机裁剪出self.scale_size宽度
    def scale_img(self,img):
        h, w = img.shape[:2]
        if h > w:
            new_h, new_w = self.scale_size * h / w, self.scale_size
        else:
            new_h, new_w = self.scale_size, self.scale_size * w / h
        
        new_h, new_w = int(new_h), int(new_w)
        img = cv2.resize(img, (new_w, new_h))
        
        if h == w:
            return img 
        elif h<w:
            top = 0
            left = np.random.randint(0, new_w - self.scale_size)
        elif h>w:
            top = np.random.randint(0, new_h - self.scale_size)
            left = 0

        img = img[top: top + self.scale_size,
                  left: left + self.scale_size]
        return img

    def preprocess(self,in_path,out_path,use_cnn=False):
        path_lst=glob.glob(in_path+'/*.jpg')
        for index, path in enumerate(path_lst):
            if index % 20==0:
                gc.collect()    # 内存清空
            print(path)
            self.split_face(path,out_path,use_cnn)
            
            
face_recognize = preprocess()
face_recognize.preprocess(in_path, out_path, use_cnn=False)