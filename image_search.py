import flickrapi
import numpy as np
import cv2
import shutil
import urllib.request
from matplotlib import pyplot as plt
api_key = u'647ef22ba1e1bc0bc6b0a7a9e96366de'
api_secret = u'989a32fc62a98f21'
flickr = flickrapi.FlickrAPI(api_key, api_secret,format='parsed-json')

def get_info(photos):
    #get url information from json file of the photos
    result = []
    for i in range(len(photos['photos'] ['photo'])): 
        image_id = photos['photos'] ['photo'][i]['id']
        size = flickr.photos.getSizes(photo_id = image_id)
        for element in size['sizes']['size']:
            if element['label'] == 'Large':
                result.append([element['source'],element['url']])
    return result

def web_crawler(result):
    #download images from url
    a = 0
    for i in result:
        a = str(a)
        url = i[0]
        file_name = a + '.jpg'
        # Download the file from `url` and save it locally under `file_name`:
        with urllib.request.urlopen(url, timeout = 10) as response, open(file_name, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
        a = int(a) + 1
        

        
def sift_compare(imgt, img0):
    grayt = cv2.cvtColor(imgt,cv2.COLOR_BGR2GRAY)
    gray0 = cv2.cvtColor(img0,cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(grayt,None)
    kp2, des2 = sift.detectAndCompute(gray0,None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k = 2)
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
    return len(good)

imgt = cv2.imread('./data/image_of_cathedral.jpg')
imgt = cv2.resize(imgt, (0,0), fx=0.25, fy=0.25)
ind = 0
result = []
flag = 0
for i in range(53,201):
    
    photos = flickr.photos.search(tags = 'Cathedral,Italy',tag_mode = 'all',per_page = 20,page = i+53)
    info = get_info(photos)
    web_crawler(info)
    for j in range(len(info)):
        a = str(j)
        img0 = cv2.imread(a + '.jpg')
        gp = sift_compare(imgt, img0)
        if gp > 300: #if enough good feature points matching
            flag += 1
            if flag > 5:
                break
                print("we may find enough answer!")
            else:
                result.append(info[j][1].replace('/sizes/l/',''))
    ind = ind + 1