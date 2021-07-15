from PIL import Image
import random
import string


def lower_size(path):
    foo = Image.open(path)
    image_size = foo.size
    x = round(image_size[0] * 0.5)
    y = round(image_size[1] * 0.5)

    foo = foo.resize((x,y),Image.ANTIALIAS)

    foo.save(path ,quality=95)
    foo.save(path ,optimize=True,quality=95)
    return path
