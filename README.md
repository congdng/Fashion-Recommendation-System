# Fashion-Recommendation-System
## Introduction

People's living conditions are now gradually improving. Thus, people start paying attention to fashion. Fashion can be seen as an aesthetic symbol of each person. People often tend to be attracted to beauty, by what attracts their eyes the most. The fashion industry has always been and will continue to evolve. Even during the recent pandemic, the fashion industry seemed to be hit hard when people were no longer able to go out. However, reality proves the opposite. Comfortable models, suitable for working, studying and practicing at home become unsurpassed.
The development of the fashion industry has pulled many trading floors, online clothing selling pages to form and appear more. Time by time, people are suffocated with the number of clothes choices and find difficulties on determining the right clothes for themselves.
Hence, this project is created. This is a system that recommends clothes based on the input of images provided by the user themself. Those images may be something that the person sees often or just glances at, but the image has attracted a certain amount of interest and that person wants to search for similar products.
The project uses neural networks to process the images from DeepFashion Dataset and k-NN to generate the final recommendations.

## Related Work

The Fashion Recommendation System has been put into application and developed at a number of online clothing websites or some applications that support building a personal style for each person. One example can be mentioned is Lookastic. When starting with this website, users will choose outfits from an existing database. From there, Lookastic will loop through the database to find images containing the clothes in the inputed inventory. It can be images from fashion magazines or images uploaded by other users publicly on the web to make outfit suggestions for users. Personal profiles are built based on visual characteristics, the visual attraction of the user. Moreover, the website also supports building a clothing schedule for each day, in accordance with user needs and local weather. Suggestions are also rated to further improve Lookastic's performance.


## Methodology

In this project, we propose a model that uses Mobilenetv3 Model, k-nearest neighbors algorithm, Yolov5, BeautifulSoup and Selenium. Initial, the Mobinetv3 model are trained with our dataset and then  another dataset containing the information and images which is crawled from some online selling website is created. When the user input searching image, that image goes through the yolov5 model to extract the clothing pieces and determine which category it belongs to. Finally, k-nearest neighbor's algorithm is used to find the most relevant products based on the input extracting image and recommendations are generated.

![Alt text](https://github.com/congdng/Fashion-Recommendation-System/blob/main/demo/Diagram.png)

## Dataset

DeepLearning2 is used for two problems. First, create a dataset to train the model to recognize which category the clothes belongs to. From that, we can derive an array (1, 2048). Second, DeepLearning2 is suitable for training yolov5 in category recognition and bounding box in an inputed image. This dataset is categorized into 13 popular categories. However, in this project, the number of categories is narrowed down to 5, including: shirt, outwear, short, skirt and dress.

https://github.com/switchablenorms/DeepFashion2

## Train the neural networks

Once the data is pre-processed, the neural networks are trained, utilizing transfer learning from Mobilenetv3. Additional layers are added in the last layers that replace the initial model to fine-tune the network model to serve the current issue.

![](https://i.imgur.com/NJ0JbBy.png)

## Data Scapping

I use BeautifulSoup and Selenium to crawl data from e-commerce sites that sell clothing related items. The information that is crawled includes the name, the url to the product, the image link, and the price.
Currently, there are two main sites being used to crawl data.
1. Amazon

Website link: amazon.com

This is a commercial website in the top of e-commerce brands.
Advantages:
- Large number of products.
- Popular all over the world.
Weakness:
- The images are often noisy, i.e. do not completely include the product image.
- The amount of data displayed and crawled is only 8 pages, about 500 products.
- Products are sometimes duplicated.

2. Darveys

Website link: amazon.com

This is a commercial website with main products are clothes for men, women and children.
Advantages:
- The number of products is quite large.
- Product images are cropped closely and with little noise components.
Weakness:
- Sometimes the image link is dead.
- The displayed currency is rupees.
- Search engine is not thorough.

## k-Nearest Neighbors algorithm

Here is a scikit-learn library, k-nearest neighbors algorithm is a classifier implementing the k-nearest neighbors vote. Feature Vectors of the same categories are trained and the trained model will predict the class for new user-input image and produce the nearest neighbors, which means the similar clothing pieces. The picture below shows how the knn works.

![](https://miro.medium.com/max/591/0*rANGSUaZLeVKnHDk.png)

## YOLOV5

YOLOV5 is the latest Object Detection model in the YOLO family of models with the same accuracy as YOLOV4 but faster prediction speed.
YOLOV5 is trained on Pytorch instead of DarkNet like previous generations. This is a big plus for YOLOv5 because PyTorch is much more popular.

## Results

Mobilenetv3 after being trained on DeepLearning 2 dataset achieved an accuracy of 74.2%. For YOLOV5 model, the accuracy achieved 76.4%. However, the recognition rate of the "dress" category is still low. One possible cause is because the images in the dataset used for training are not good and a large number of images contain "dress" accompanying the wearer's legs. Thus, when identifying, the model often relies on this feature to identify dress. Below is the confusion matrix of the YOLOV5 model.

![](https://github.com/congdng/Fashion-Recommendation-System/blob/main/yolov5/runs/train/Model11/confusion_matrix.png)

## Screenshots

<h3>Main screen of the website</h3>

![](https://i.imgur.com/cuG1osH.png)

User upload searching image to the website. Then, the YOLOV5 model extract clothing pieces and show below.

<h3>Option screen</h3>

![](https://i.imgur.com/GvVQJVj.png)

User choose the number according to the extracting image. Then, knn determine which images in the database which are very similar to the extracting image and show.

## Installation

Use pip to install the requirements.

~~~bash
pip install -r requirements.txt
~~~
  
## Usage
Follow these below step:
- Download the DeepLearning2 Dataset. After downloading the dataset, three folders including train, validation and test are formed. Each folder has two another smaller folders: annos containing seperated json file for each image; image.
- Run the json_combine.py to combine all the json file of the trainset and valid set.
- Run the category_change.py. Since the initial dataset has 13 categories, we need to narrow down to 5 categories.
  + short sleeve top, long sleeve top, vest, sling: shirt.
  + short sleeve outwear, long sleeve outwear: outwear.
  + shorts, trousers: short.
  + skirt. 
  + short sleeve dress, long sleeve dress, vest dress, sling dress: dress.
- Run cut_image.py to extract the clothing pieces from the dataset's image and store it to a file for training later.
- Run mobinet_model.py to set up the CNN pre-trained model, Mobilenetv3 and train it on our dataset. Currently, my model achieves 74.2% in accuracy. In this file, we can also plot the diagram which shows the accuracy and loss's movement through epochs.
- If the model's accuracy is not high enough to satisfy our demand, run the model_evaluation.py to keep training the model.
- Since YOLOV5 need specified format of file, run the yolo_train_dataset.py and yolo_valid_dataset.py to set up label.txt for each image. The format of this file can be seen as below.

![](https://i.imgur.com/1wafJgv.png)

- Run crawl_data.py to crawl the data from some website on the Internet. Currently, I crawl from amazon.com and darveys.com.
- Run download_image.py 
- Run get_image_numarray.py. This file let the trained Mobilenetv3 model run on the crawl dataset and extract a vector array with size (1,2048) for each image. The result then is stored to a json file.
- Run knnfile.py to apply k-Nearest Neighbors algorithm on the crawl dataset. After running this, a file containing knn infomation is created in each category file.
- Run crawl_seperate.py to make seperate JSON file for each image. This helps speed up the performance of the website.
- Finally, run the web server, enter below:

```bash
streamlit run main.py
```

## Library

- [OpenCV]()
- [Tensorflow]()
- [streamlit]()
- [scikit-learn]()

## Conclusion

In this project, we have completed a simple and effective recommendation system that has real and direct links to e-commerce websites that deal in fashion products.
The system wants to find the corresponding recommendations to go through stages. The first is training a CNN pre-trained model which is Mobilenetv3 to an accuracy of 74.2%. This is definitely something to improve on in the future. A vector array (1, 2048) is extracted for classification through the knn algorithm combined with extracting inputed images from the results of the YOLOV5 model.
This project is still in development. Expand more categories such as western shoes, sneakers, accessories, jewelry, etc. Format and perform current functions even when the user uses his real image through the webcam. Add other fashion styles by crawling more information from many websites from different countries.

## Credit

This "Fashion Recommendation System" project is not only the result of my own constant efforts, but also the support and encouragement of my instructor Huynh Nguyen Minh Thong. I would like to give special thanks to Mr.Huynh Nguyen Minh Thong - who helped me during my study - research for this project.
