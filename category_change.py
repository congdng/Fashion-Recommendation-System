import json

TRAINNO = 162331
VALIDNO = 52199
TRAINFOLDER = 'train/annos/'
VALIDFOLDER = 'validation/annos/'
TRAINJSONPATH = 'sum.json'
VALIDJSONPATH = 'validsum.json'

#Since the dataset contains 13 categories, we need to narrow down to 5 categories: shirt, outwear, short, skirt and dress
def change_category(FOLDERPATH, JSONPATH, num_of_images):
    with open(FOLDERPATH, 'r') as f:
        temp = json.loads(f.read())
        for i in temp:
            if i == 'item':
                continue
            else:
                for s in range(num_of_images):
                    if (temp[i][s]['categories'] == 1) | (temp[i][s]['categories'] == 2) | (temp[i][s]['categories'] == 5) | (temp[i][s]['categories'] == 6):
                        temp[i][s]['categories'] = 'shirt'
                    elif (temp[i][s]['categories'] == 3) | (temp[i][s]['categories'] == 4):
                        temp[i][s]['categories'] = 'outwear'
                    elif (temp[i][s]['categories'] == 7) | (temp[i][s]['categories'] == 8):
                        temp[i][s]['categories'] = 'short'
                    elif (temp[i][s]['categories'] == 9):
                        temp[i][s]['categories'] = 'skirt'
                    else:
                        temp[i][s]['categories'] = 'dress'
    json_name = JSONPATH
    with open(json_name, 'w') as f:
        json.dump(temp, f)
        print('Done')

def main():
    change_category(TRAINJSONPATH, TRAINJSONPATH, TRAINNO)
    change_category(VALIDJSONPATH, VALIDJSONPATH, VALIDNO)

if __name__ == '__main__':
    main()
