import cv2 
import pytesseract
from pytesseract import Output
import re


from preprocessing import *


img_fname = 'eng_bw.png'
img = cv2.imread(img_fname)
print('loaded image')



# Adding custom options
custom_config = r'--oem 3 --psm 6'


#text output from image to string
output = pytesseract.image_to_string(img, config=custom_config)
print(output)

#check preprocessing outputs
# image = cv2.imread('aurebesh.jpg')

gray = get_grayscale(img)
cv2.imwrite('./gray.png',gray)

thresh = thresholding(gray)
cv2.imwrite('./thresh.png',thresh)

opening = opening(gray)
cv2.imwrite('./opening.png',opening)

canny = canny(gray)
cv2.imwrite('./canny.png',canny)


#get the bounding box around text (character level)

h, w, c = img.shape
boxes = pytesseract.image_to_boxes(img) 
for b in boxes.splitlines():
    b = b.split(' ')
    img_with_char_box = cv2.rectangle(img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)

pytesseract_bounding_boxes_fname = './bounding_boxes_charwise.png'
cv2.imwrite(pytesseract_bounding_boxes_fname,img_with_char_box)
# cv2.imshow('img', img)
# cv2.waitKey(0)



img = cv2.imread(img_fname)

d = pytesseract.image_to_data(img, output_type=Output.DICT)
print(d.keys())

#dict_keys(['level', 'page_num', 'block_num', 'par_num', 'line_num', 'word_num', 'left', 'top', 'width', 'height', 'conf', 'text'])

#use this to get the box around each word
n_boxes = len(d['text'])
for i in range(n_boxes):
    if int(d['conf'][i]) > 60:
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        img_with_word_box = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


pytesseract_bounding_boxes_fname = './bounding_boxes_wordwise.png'
cv2.imwrite(pytesseract_bounding_boxes_fname,img_with_word_box)



## pattern matching, currently matches with date
img = cv2.imread(img_fname)
d = pytesseract.image_to_data(img, output_type=Output.DICT)
keys = list(d.keys())

date_pattern = '^(0[1-9]|[12][0-9]|3[01])/(0[1-9]|1[012])/(19|20)\d\d$'

n_boxes = len(d['text'])
for i in range(n_boxes):
    if int(d['conf'][i]) > 60:
    	if re.match(date_pattern, d['text'][i]):
	        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
	        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow('img', img)
cv2.waitKey(0)
