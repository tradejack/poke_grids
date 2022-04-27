from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
import math
import requests
import io
import time

############################
#  Define Color Constants  #
############################

black_square  = "\U00002B1B"
green_square  = "\U0001F7E9"
yellow_square = "\U0001F7E8"
white_square  = "\U00002B1C"
red_square    = "\U0001F7E5"
orange_square = "\U0001F7E7"
blue_square   = "\U0001F7E6"
purple_square = "\U0001F7EA"
brown_square  = "\U0001F7EB"
clear_square  = "  "
#U0001F532 
squares = [black_square, green_square, yellow_square, white_square, red_square, orange_square, blue_square, purple_square, brown_square, clear_square, clear_square]

BLACK  = (70, 80, 70, 255)
GREEN  = (113, 235, 60, 255)
YELLOW = (255, 235, 100, 255)
WHITE  = (255, 255, 255, 255)
RED    = (244, 80, 7, 255)
ORANGE = (245, 170, 50, 255)
BLUE   = (13, 100, 219, 255)
PURPLE = (210, 20, 210, 255)
BROWN  = (140, 100, 50, 255)
CLEAR  = (0, 0, 0, 0)
CLEAR2 = (255,255,255,0)
colors = [BLACK, GREEN, YELLOW, WHITE, RED, ORANGE, BLUE, PURPLE, BROWN, CLEAR, CLEAR2]

def square_crop(img, verbose=False):
    """
    takes a pillow imagey object
    returns the smallest square that cuts out the most whitespace
    """
    pix = img.load()
    width, height = img.size

    # first we want to compute the margins of whitespace in each direction
    # we do a horizontal and vertical pass to iterate in different directions
    # possible to optimize with one for loop if you track a lot more things
    # but i don't want to do that :shrug:

    # top-bottom pass
    found_X = 0 # set flags to track when image is detected along axes
    for y in range(height):
        seen_color = False
        for x in range(width):
            color = pix[x,y]
            if color[3] != 0:
                seen_color = True
                if found_X == 0:
                    top = y
                    found_X = 1
        if found_X == 1 and not seen_color:
            found_X = 2
            bottom = height - y

    # left-right pass
    found_Y = 0 
    for x in range(width):
        seen_color = False
        for y in range(height):
            color = pix[x,y]
            if color[3] != 0:
                seen_color = True
                if found_Y == 0:
                    left = x
                    found_Y = 1
        if found_Y == 1 and not seen_color:
            found_Y = 2
            right = width - x
    
    if verbose:
        print("XY Flag values: ", found_X, found_Y)

    if found_X != 2 or found_Y != 2:
        # if either of the flags hasn't ticked twice
        # then one of the directions' margins in 0
        # and we're already done
        return img

    if verbose:
        print(f"Top: {top}\nBottom: {bottom}\nLeft: {left}\nRight: {right}")

    margin = min(top, bottom, left, right)
    return img.crop((margin, margin, width-margin, height-margin))

def unroll(image):
    w, h, d  = tuple(image.shape)
    assert d == 4
    image_array = np.reshape(image, (w * h, d))
    return image_array
def reroll(image):
    l = tuple(image.shape)[0]
    image_array = np.reshape(image, (int(math.sqrt(l)), int(math.sqrt(l))))
    return image_array

def kmeans_color(chunk, kmeans, n_colors = 5):
    labels = kmeans.predict(unroll(chunk))
    counts = Counter(labels)
    max_index = counts.most_common()[0][0]
    return np.rint(abs(kmeans.cluster_centers_[max_index]))

def nearest_color(chunk_color):
    min_dist = 255*255
    min_dex  = 0
    a = np.array(chunk_color)
    for i,color in enumerate(colors):
        b = np.array(color)
        dist = np.linalg.norm(a-b)
        if dist < min_dist:
            min_dist = dist
            min_dex = i
    return squares[min_dex]

def generate_squares(file, splits, n_colors, image_debug = False, filename="debug"):
    img = Image.open(file)
    img = img.convert('RGBA')
    pix_arr = np.array(square_crop(img))

    chunks = []
    for n in np.array_split(pix_arr, splits):
        chunks.append(np.array_split(n, splits, axis = 1))
        
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(unroll(pix_arr))
    col_grid = []
    for row in chunks:
        row_blocks = ""
        col_blocks = []
        for chunk in row:
            c = kmeans_color(chunk, kmeans)
            col_blocks += [c]
            row_blocks += nearest_color(c)
        col_grid += [col_blocks]
        print(row_blocks)

    if image_debug:
        mult = 13
        col_grid = np.array(col_grid, dtype=np.uint8).repeat(mult, axis=0).repeat(mult, axis=1)[:, :, 0:3]
        print(col_grid.shape)
        img = Image.fromarray(col_grid, 'RGB')
        img.save(filename+".png")
        #img.show()
    
def get_pokemon(id, s=11,c=11, filename="debug"):
    url = 'https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/'+str(id)+'.png'
    r = requests.get(url)
    generate_squares(io.BytesIO(r.content), s, c, image_debug=True, filename=filename)

# for i in range(20, 40):
#     print(i)
#     get_pokemon(i, 11, 8, filename="debug"+str(i))
#     print()

get_pokemon(722, 11, 8)

api_key = "Jl9KUPYgHcj7OeFXAjmTvie9H"
api_key_secret = "iYEXDon1biU6rkanWmJYCh2bWHavUBtny3NYE2MPxhHmBLlwJP"
bearer = "AAAAAAAAAAAAAAAAAAAAAHwXYAEAAAAA6ZeeACyI2RpZtLH%2B5PJG6XAUT7U%3DEtTSplGg44IekjJcyufwX0lAU6SkYqT4lDnR0fqJoWhkP63Ya0"