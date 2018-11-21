# Fine-grained-Sentiment-Analysis-of-User-Reviews
AI challenger, Challenge multi-level and multi-dimensional fine-grained sentiment analysis and seek more accurate algorithms to predict users' fine-grained sentimental tendencies 

## Introduction

Online reviews are becoming the critical factor to make consumption decision in recent years. They not only have a profound impact on the incisive understanding of shops, users, and the implied sentiment, but also have been widely used in Internet and e-commerce industry, such as personalized recommendation, intelligent search, product feedback, and business security. In this challenge, we provide a high quality mass dataset, including a total of 6 categories of 20 fine-grained elements of emotional sentiment. Participants are required to establish an algorithm based on the sentimental tendency of the fine-grained elements, and perform sentiment mining on user comments. The organizing committee will determine the prediction accuracy rate by calculating the difference between the participant's submitted prediction value and the actual value, thus evaluate the submitted prediction algorithm.

## Data

The dataset is divided into four parts: training, validation, test A and test B.This dataset builds a two-layer labeling system according to the evaluation granularity:  the first layer is the coarse-grained evaluation object, such as “service” and “location”; the second layer is the fine-grained emotion object, such as “waiter’s attitude” and “wait time” in “service” category. The specific description is shown in the following table. 


The first layer  | The second layer
  ------------- | -------------
 location |	traffic convenience
||distance from business district
||easy to find
service |	wait time
||waiter’s attitude
||parking convenience
||serving speed
price |	price level
||cost-effective
||discount
environment |	decoration
||noise
||space
||cleaness
dish |	portion
||taste
||look
||recommendation
others |	overall experience
||willing to consume again 

There are four sentimental types for every fine-grained element: Positive, Neutral, Negative and Not mentioned, which are labelled as 1, 0, -1 and-2. The meaning of these four labels are listed below.

 Sentimental labels |	1  |	0  |	-1  |	-2
------------- | -------------| -------------| -------------| -------------
Meaning  |	Positive  |	Neutral  |	Negative | 	Not mentioned
An example of one labelled review:
    “味道不错的面馆，性价比也相当之高，分量很足～女生吃小份，胃口小的，可能吃不完呢。环境在面馆来说算是好的，至少看上去堂子很亮，也比较干净，一般苍蝇馆子还是比不上这个卫生状况的。中午饭点的时候，人很多，人行道上也是要坐满的，隔壁的冒菜馆子，据说是一家，有时候也会开放出来坐吃面的人。”
 The first layer |	The second layer |	Label
------------- | -------------| -------------
location |	traffic convenience |	-2
||distance from business district |	-2
||easy to find |	-2
service |	wait time |	-2
||waiter’s attitude |	-2
||parking convenience |	-2
||serving speed |	-2
price |	price level |	-2
||cost-effective |	1
||discount |	-2
environment |	decoration |	1
||noise |	-2
||space |	-2
||cleaness |	1
dish |	portion |	1
||taste |	1
||look |	-2
||recommendation |	-2
others |	overall experience |	1
||willing to consume again |	-2 

## Submission
The participants need to predict the sentimental tendency of the 6 categories of 20 fine-grained elements on the test set according to the trained model, and submit the prediction result. The prediction result is described by four values of [-2,-1,0,1]. The result needs to be saved as a csv file. The format is as follows: 
 
  id |	content |	location_traffic_ convenience |	location_distance_ from_business_district |	location_easy_to_find |...
 ------------- | ------------- | ------------- | ------------- | ------------- | -------------
 ||||||
 
 
Label field description:

 .|.|.|.
------------- | ------------- | ------------- | -------------
location_traffic_ convenience | location_distance_from_ business_district |	location_easy_ to_find |	service_ wait_time
service_waiters_ attitude |	service_parking_ convenience |	service_serving_ speed |	environment_noise
price_cost_ effective |	price_discount |	environment_ decoration |	environment_noise
environment_space 	|environment_ cleaness |	dish_portion |	dish_taste
dish_look |	dish_recommendation |	others_overall_ experience |	others_willing_to_ consume_again
 
## Embedding
To download the chinese word vectors trained with different representations, context features, and corpora.
https://github.com/Embedding/Chinese-Word-Vectors
In this project we use sgns.zhihu.bigram.
