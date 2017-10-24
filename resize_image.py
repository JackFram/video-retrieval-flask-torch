import os
from PIL import Image
#reference to yunjey
def resize_image(image,size):
    '''resize an image to a specific size'''
    return image.resize(size, Image.ANTIALIAS)
def resize_images(image_dir, output_dir, size):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for index in range(10):
        dir=image_dir+'/image_data_'+str(index+1)
        images=os.listdir(dir)
        num_images=len(images)
        for i, image in enumerate(images):
            try:
                with open(os.path.join(dir, image), 'r+b') as f:
                    with Image.open(f) as img:
                        img = resize_image(img, size)
                        img_path = str(index+1) + '_' + image
                        img.save(os.path.join(output_dir, img_path), img.format)
                if(i%100==0):
                    print("%d : [%d/%d] Resized the images and saved into '%s'." % (index, i, num_images, output_dir))
            except:
                continue
# resize_images('/Users/zhangzhihao/Documents/webbrain/data/images','/Users/zhangzhihao/Documents/webbrain/data/resized_images',(256,256))