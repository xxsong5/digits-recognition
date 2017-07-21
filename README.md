# image Distis
This is the code for number(0-9) recognition in image by using SVM Classifier and HOG descripter. And the main algorithm was packed as classes in imgDistis.cpp 


## how to combile  
mkdir build  
cd build  
cmake ..  
make  


## how to use example there need three steps for running this code. 
1.  save number image to a directory  and Format("img%05d_d.png") should be used as image name, the second 'd' in the format must be (0-9) asscociated with number's image or 11 for noise image.
for example:  ./saveNUMImgs  .../images-path 
Note. the zero should be named (0-9 ,11) asscociated with the image for training. ("img00001\_0.png-> img00001\_(0-9,11).png")

2.  using the number-images to train number's-model.xml 
for example:  ./train .../number-images-path

3. using the trained number's-model.xml to detect new image
for example: ./detectNUMs  .../new-image  (.../number's-model.xml, can be ignored, because there is one in trainData directory)


## exmaple
1.  ./saveNUMImgs ./character_0_1_2_3/  
note. Press 's' key for saving numbers' images to (./trainData/) directory.

2.  ./train ./trainData/
note. Before run this commond, make sure you have changed The Zero in image name to associated number.

3. ./detectNUMs  ./new-image  ./trainData/number_model.xml 
note. The second param can be ignored if there is a number_model.xml file in trainData directory. For sample use: ./detectNUMs  ./new-image


## currently
 there is a partial trained images model. For test, one can just run the third step (./detectNUMs  ./new-image).


##Notation
By the way, this project just fit our purpose,  One must rewrite the segment(..) function your own project to generate segmented numbered images for trainnig.

