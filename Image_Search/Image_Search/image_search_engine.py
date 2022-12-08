
from tkinter.messagebox import NO
from articles import articles as keywords
from image_download import *
from shutil import rmtree
from threading import Thread

import os
import time

import numpy as np
import cv2 as cv
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import pickle

# img = cv.imread('images/000001.png')

# sift = cv.SIFT_create()
# kp = sift.detect(img,None)
# img=cv.drawKeypoints(img,kp,img)
# cv.imwrite('sift_keypoints.jpg',img)



class ImageSearchEngine():
    def __init__(self, kmeans_num_clusters=50) -> None:
        self.kmeans_num_clusters = kmeans_num_clusters
        self.SIFT = cv.xfeatures2d.SIFT_create()
    
    def build(self, keywords):
        
        self.downloadImages(keywords)
        self.keywords = keywords
        
        all_descriptors = []
        img_features = []

        for img_name in os.listdir('images'):
            img = cv.imread(f'images/{img_name}')
            _, descriptors = self.SIFT.detectAndCompute(img, None)
            if descriptors is not None:
                for des in descriptors:
                    all_descriptors.append(des)
            img_features.append(descriptors)

        self.kmeans = KMeans(n_clusters = self.kmeans_num_clusters, n_init=10)
        st = time.time()
        self.kmeans.fit(all_descriptors)
        print(f"Kmeans training took {time.time() - st} s....")

        visual_words = self.kmeans.cluster_centers_


        self.imageVectors = []

        for descriptors in img_features:
            self.imageVectors.append(self.getImageVectorFromDescriptors(descriptors))

        
    def getImageVectorFromDescriptors(self, descriptors):
        hist = np.zeros(self.kmeans_num_clusters)
        if descriptors is None:
            return hist
        descriptors = np.array(descriptors, dtype=float)
        classes = self.kmeans.predict(descriptors)
        for class_idx in classes:
            hist[class_idx] += 1.0
        return hist


    def getImageVector(self, img):
        _, descriptors = self.SIFT.detectAndCompute(img, None)
        return self.getImageVectorFromDescriptors(descriptors)

    # Expects a PIL image
    def getSearchResults(self, img, max_results=5):
        img = preProcessImage(img)
        img.save('tmp.jpg')
        img = cv.imread('tmp.jpg')
        query_vec = self.getImageVector(img)
        query_vec = sparse.csr_matrix(query_vec)
        results = cosine_similarity(self.imageVectors, query_vec).reshape((-1,))
        ret = []
        for idx in results.argsort()[::-1][:max_results]:
            ret.append(f'images/{self.keywords[idx]}.jpg')
        os.remove('tmp.jpg')
        return ret




    def save(self, save_file_name='img_se.obj'):
        f = open(save_file_name, 'wb')
        pickle.dump((self.kmeans_num_clusters, self.keywords, self.imageVectors, self.kmeans), f)
        f.close()

    def load(self, load_file_name='img_se.obj'):
        f = open(load_file_name, 'rb')
        obj = pickle.load(f)
        self.kmeans_num_clusters = obj[0]
        self.keywords = obj[1]
        self.imageVectors = obj[2]
        self.kmeans = obj[3]
        f.close()

    

    
    def downloadImages(self, keywords):
        img_dir = os.path.join(os.getcwd(), 'images')
        try:
            rmtree(img_dir)
        except:
            pass
        
        threads = []
        for idx, word in enumerate(keywords):
            #getImage(word, idx)
            th = Thread(target=getImage, args=(word, idx))
            threads.append(th)
            th.start()
           
        for th in threads:
            th.join()
        resizeAndGreyOutImages(keywords)





# im_se = ImageSearchEngine()
# im_se.build(keywords[:10]) 

# im_se.save()

# im_se = ImageSearchEngine()
# im_se.load()

# img = Image.open('tay.jpg')

# res = im_se.getSearchResults(img)

# print(res)




