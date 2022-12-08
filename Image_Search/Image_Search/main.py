import requests
from PIL import Image
from image_search_engine import ImageSearchEngine
import os
import time

from articles import articles as keywords

def clear_screen():
    if os.name == 'nt':
        _ = os.system('cls')
  
    else:
        _ = os.system('clear')




clear_screen()

load = False
kmeans_num_clusters = 50
im_se = ImageSearchEngine(kmeans_num_clusters)

if not load:
    im_se.build(keywords[:]) 
    im_se.save()
else:
    im_se.load()


clear_screen()

while True:
    cmd = input('Enter surl to search from image url, sfile to search from file, exit to exit...\n').strip()
    if cmd == 'exit':
        exit(0)
    if cmd == 'surl':
        url = input('Enter url...\n').strip()
        img = Image.open(requests.get(url, stream=True).raw)

    if cmd == 'sfile':
         file = input('Enter file path...\n').strip()
         img = Image.open(file)
    else:
        print('Invalid command')
        continue

    st = time.time()
    num_results = 5
    results = im_se.getSearchResults(img, num_results)
    print(f'Search took {time.time() - st} s...')
    for idx, res in enumerate(results):
        print(f'[{idx+1}/{num_results}] {res}')
        img = Image.open(res)
        img.show()
        if idx + 1 < num_results:
            cmd = input("Enter next to show next result, anything else to search again\n")
            if cmd == 'next':
                continue
            else:
                break









