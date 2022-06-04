import json

TRAINFOLDER = 'train/annos/'
VALIDFOLDER = 'validation/annos/'
TRAINJSONPATH = 'sum.json'
VALIDJSONPATH = 'validsum.json'


#Use to combine all the json in the DeepLearning Dataset to one main json file
#Take 32000 images from validation file and 100000 images from train file

traindataset = {
    "item": {},
    "info": []
}

validdataset ={
    "item": {},
    "info": []
}

def combine_json(FOLDERPATH, num_of_images, datadict):
    j = 1
    for num in range(num_of_images):
        name = FOLDERPATH + str(num).zfill(6) + '.json'
        imagename = str(num).zfill(6) + '.jpg'
        if (num > 0):
            with open(name, 'r') as f:
                temp = json.loads(f.read())
                #print(temp)
                for i in temp:
                    if i == 'source' or i=='pair_id':
                        continue
                    else:
                        box = temp[i]['bounding_box']
                        bbox=[box[0],box[1],box[2],box[3]]
                        cat = temp[i]['category_id']
                        datadict['info'].append({
                            "no": j,
                            "categories": cat,
                            "boundingbox": bbox,
                            "image": imagename,
                        })
                    j += 1
    print(len(datadict['info']))

def exportjson(JSONPATH, datadict):
    json_name = JSONPATH
    with open(json_name, 'w') as f:
        json.dump(datadict, f)
        print('Data extracted')

def main():
    combine_json(TRAINFOLDER, 100000, traindataset)
    combine_json(VALIDFOLDER, 32000, validdataset)
    exportjson(TRAINJSONPATH, traindataset)
    exportjson(VALIDJSONPATH, validdataset)

if __name__ == '__main__':
    main()
