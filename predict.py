
from yolo import YOLO
from PIL import Image
import os


path = 'C:/Users/user/Desktop/test_pics/out_2/'
yolo = YOLO()

i = 0

for img in os.listdir(path):
    # i = files.index(img)
    i += 1
    try:
        image = Image.open(path+img)
    except:
        print('Open Error! Try again!')
        continue
    else:
        r_image = yolo.detect_image(image)
        # r_image.show()
        r_image.save("C:/Users/user/Desktop/test_pics/Ablation/"+ "SPP-100/" + "{}.jpg".format(i))






# while True:
#     img = input('Input image filename:')
#     try:
#         image = Image.open(img)
#     except:
#         print('Open Error! Try again!')
#         continue
#     else:
#         r_image = yolo.detect_image(image)
#         r_image.show()
