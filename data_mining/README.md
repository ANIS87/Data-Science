This project is a simple GUI developed into  python 2.7. 
The goal of this tool is  to simplify the data mining collected from analytics database. 
This allows  to test  the random forest  algorithm with  the data given by the shop_match file,  to  perform an unsupervised clustring of global data and to make some statistical analysis ( e.g., display the histogram of a given  collection).  

#######    recommandations ########

Before testing this tool, please make sure you have the following libraries:

 *   Tkinter: for GUI

 *   PIL: image processing 

 *   pandas: statistical analysis

 *   sklearn: statistical analysis

 *   kmodes: pip install kmodes from https://github.com/nicodv/kmodes

 *   shop_match.bson : the data file. 

#######  Usage  ########

python TOOL_MINING_VERSION_ALPHA_KRAYNI.py # This script  is the main file.


#######  Explanations ########
The "data" button is to transform a BSON file ( given by mongodb) into a csv file. The "machine learning" button is to test algorithms like random forest and to make the sensitivity analysis (e.g., study the impact of such  parameter on the final  model).
The button "clustering" is to test the classification algorithm "K-prototypes." The statistics button is to display histograms (number of click / view by each city), cpc for each city .... Each button has various options (menu button).

![tool_data_shopedia.png](https://bitbucket.org/repo/4k49eB/images/1971819329-tool_data_shopedia.png)

The help button can display a paragraph explaining the steps followed in each algorithm.