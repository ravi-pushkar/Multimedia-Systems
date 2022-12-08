import os
from PIL import Image, ImageOps
import math


from icrawler.builtin import GoogleImageCrawler




# TODO: Use guassian blurring

def getImage(keyword, file_no):
    google_Crawler = GoogleImageCrawler()
    google_Crawler.crawl(keyword, max_num=1, file_idx_offset=file_no)


# expects a PIL image
def preProcessImage(img):
    (h, w) = img.size
    asr = h / w
    if max(h, w) > 1000:
        if h >= w:
            h, w = 1000, math.floor(1000 / asr)
        else:
            h, w = math.floor(1000 * asr) , 1000
        img.resize((h, w))
    img = ImageOps.grayscale(img)
    return img

def resizeAndGreyOutImages(keywords):
    for img_name in os.listdir('images/'):
        img = Image.open(f'images/{img_name}')
        img = preProcessImage(img)
        os.remove(f'images/{img_name}')
        [img_no, extension] = img_name.split('.')
        #img_name = keywords[int(img_no)-1] + '.' + extension
        img_name = keywords[int(img_no)-1] + '.' + 'jpg'
        img.save(f'images/{img_name}')





"""Tests"""

# getImage('taylor swift')

# resizeAndGreyOutImages()


