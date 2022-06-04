import cv2
import os 
import json

TRAINNO = 162331
VALIDNO = 52199
TRAINFOLDER = 'train/image/'
VALIDFOLDER = 'validation/image/'
TRAINJSONPATH = 'sum.json'
VALIDJSONPATH = 'validsum.json'
TRAINSAVEFOLDER = 'traindata/'
VALIDSAVEFOLDER = 'validdata/'

#Deploy images from the DeepFashion dataset for our training
#Take 11250 images, each categories has 2250 images for validation
#Take 37500 images, each categories has 7500 images for training

def cut_image(FOLDERSAVEPATH, JSONPATH, num_of_images, num_of_category_images):
    with open(JSONPATH, 'r') as f:
        temp = json.loads(f.read())
        for i in temp:
            if i == 'item':
                continue
            else:
                for s in range(num_of_images):
                    if (len(os.listdir(FOLDERSAVEPATH + str(temp[i][s]['categories']))) < num_of_category_images):
                        image = cv2.imread(FOLDERSAVEPATH + '/'+ temp[i][s]['image'])
                        cv2.imwrite(FOLDERSAVEPATH + str(temp[i][s]['categories']) + '/' + str(temp[i][s]['no']).zfill(6) + '.jpg', image[temp[i][s]['boundingbox'][1]:temp[i][s]['boundingbox'][3], temp[i][s]['boundingbox'][0]:temp[i][s]['boundingbox'][2]])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    cut_image(TRAINSAVEFOLDER, TRAINJSONPATH, TRAINNO, 7500)
    cut_image(VALIDSAVEFOLDER, VALIDJSONPATH, VALIDNO, 2250)

if __name__ == '__main__':
    main()