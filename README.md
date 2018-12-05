# Fine_grained_Sentiment_Analysis_of_User_Reviews
# 细粒度用户评论情感分析
AI challenger, Challenge multi-level and multi-dimensional fine-grained sentiment analysis and seek more accurate algorithms to predict users' fine-grained sentimental tendencies
挑战多层次、多维度的细粒度情感分析，寻求更精准的算法预测用户的细粒度情感倾向

## Introduction
## 简介
Online reviews are becoming the critical factor to make consumption decision in recent years. They not only have a profound impact on the incisive understanding of shops, users, and the implied sentiment, but also have been widely used in Internet and e-commerce industry, such as personalized recommendation, intelligent search, product feedback, and business security. In this challenge, we provide a high quality mass dataset, including a total of 6 categories of 20 fine-grained elements of emotional sentiment. Participants are required to establish an algorithm based on the sentimental tendency of the fine-grained elements, and perform sentiment mining on user comments. The organizing committee will determine the prediction accuracy rate by calculating the difference between the participant's submitted prediction value and the actual value, thus evaluate the submitted prediction algorithm.  
在线评论的细粒度情感分析对于深刻理解商家和用户、挖掘用户情感等方面有至关重要的价值，并且在互联网行业有极其广泛的应用，主要用于个性化推荐、智能搜索、产品反馈、业务安全等。本次比赛我们提供了一个高质量的海量数据集，共包含6大类20个细粒度要素的情感倾向。参赛人员需根据标注的细粒度要素的情感倾向建立算法，对用户评论进行情感挖掘，组委将通过计算参赛者提交预测值和场景真实值之间的误差确定预测正确率，评估所提交的预测算法。
## Data
## 数据说明
The dataset is divided into four parts: training, validation, test A and test B.This dataset builds a two-layer labeling system according to the evaluation granularity:  the first layer is the coarse-grained evaluation object, such as “service” and “location”; the second layer is the fine-grained emotion object, such as “waiter’s attitude” and “wait time” in “service” category. The specific description is shown in the following table.  
数据集分为训练、验证、测试A与测试B四部分。数据集中的评价对象按照粒度不同划分为两个层次，层次一为粗粒度的评价对象，例如评论文本中涉及的服务、位置等要素；层次二为细粒度的情感对象，例如“服务”属性中的“服务人员态度”、“排队等候时间”等细粒度要素。评价对象的具体划分如下表所示。  

层次一(The first layer) |	层次二(The second layer)
  ------------- | -------------
位置(location) |	交通是否便利(traffic convenience)
||距离商圈远近(distance from business district)
||是否容易寻找(easy to find)
服务(service) |	排队等候时间(wait time)
||服务人员态度(waiter’s attitude)
||是否容易停车(parking convenience)
||点菜/上菜速度(serving speed)
价格(price) |	价格水平(price level)
||性价比(cost-effective)
||折扣力度(discount)
环境(environment) |	装修情况(decoration)
||嘈杂情况(noise)
||就餐空间(space)
||卫生情况(cleaness)
菜品(dish) |	分量(portion)
||口感(taste)
||外观(look)
||推荐程度(recommendation)
其他(others) |	本次消费感受(overall experience)
||再次消费的意愿(willing to consume again) 


There are four sentimental types for every fine-grained element: Positive, Neutral, Negative and Not mentioned, which are labelled as 1, 0, -1 and-2. The meaning of these four labels are listed below.  
每个细粒度要素的情感倾向有四种状态：正向、中性、负向、未提及。使用[1,0,-1,-2]四个值对情感倾向进行描述，情感倾向值及其含义对照表如下所示：  

 情感倾向值(Sentimental labels) |	1  |	0  |	-1  |	-2
------------- | -------------| -------------| -------------| -------------
含义（Meaning） |	正面情感(Positive)	| 中性情感(Neutral) |	负面情感（Negative）| 	情感倾向未提及（Not mentioned）  

An example of one labelled review:  
数据标注示例如下： 
```
    “味道不错的面馆，性价比也相当之高，分量很足～女生吃小份，胃口小的，可能吃不完呢。环境在面馆来说算是好的，至少看上去堂子很亮，也比较干净，一般苍蝇馆子还是比不上这个卫生状况的。中午饭点的时候，人很多，人行道上也是要坐满的，隔壁的冒菜馆子，据说是一家，有时候也会开放出来坐吃面的人。”
```
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
## 提交结果说明
The participants need to predict the sentimental tendency of the 6 categories of 20 fine-grained elements on the test set according to the trained model, and submit the prediction result. The prediction result is described by four values of [-2,-1,0,1]. The result needs to be saved as a csv file. The format is as follows:   
选手需根据训练的模型对测试集的6大类20个的细粒度要素的情感倾向进行预测，提交预测结果，预测结果使用[-2,-1,0,1]四个值进行描述，返回的结果需保存为csv文件。格式如下：

  id |	content |	location_traffic_ convenience |	location_distance_ from_business_district |	location_easy_to_find |...
 ------------- | ------------- | ------------- | ------------- | ------------- | -------------
 |||||| 
  
Label field description:  
标注字段说明：  

 .|.|.|.
------------- | ------------- | ------------- | -------------
location_traffic_ convenience |	location_distance_from_ business_district |	location_easy_ to_find |	service_ wait_time
位置-交通是否便利 |	位置-距离商圈远近 |	位置-是否容易寻找 |	服务-排队等候时间
service_waiters_ attitude |	service_parking_ convenience |	service_serving_ speed | environment_noise
服务-服务人员态度 |	服务-是否容易停车 |	服务-点菜/上菜速度 |	价格-价格水平
price_cost_ effective |	price_discount |	environment_ decoration |	environment_noise
价格-性价比 |	价格-折扣力度 |	环境-装修情况 |	环境-嘈杂情况
environment_space |	environment_ cleaness |	dish_portion |	dish_taste
环境-就餐空间 |	环境-卫生情况 |	菜品-分量 |	菜品-口感
dish_look |	dish_recommendation |	others_overall_ experience |	others_willing_to_ consume_again
菜品-外观 |	菜品-推荐程度 |	其他-本次消费感受 |	其他-再次消费的意愿

## Embedding
## 词嵌入
To download the chinese word vectors trained with different representations, context features, and corpora.
https://github.com/Embedding/Chinese-Word-Vectors
In this project we use sgns.zhihu.bigram.  
[作者所用词向量：sgns.zhihu.bigram](https://github.com/Embedding/Chinese-Word-Vectors)   
### PS 
### 注
Cuz the dataset of AI challenger is too large to upload, three simplified datasets are applied in this demo.  
在案例中给予了小样本测试集（比赛非完整数据集）。
