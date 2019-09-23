# Identify-Number-of-People-Available-in-a-Location-Based-on-WiFi-RSSI-Data

## Project Description

WiFi-RSSI data obtained from different Access Points using a NodeMCU (ESP8266) are published to a MQTT server.
Those data are then collected and fed to a classification model to predict the number of people available in that location.

Collected data are classfied under following 3 classes:
  A. Empty area vs. one person present and moving about.
  B. Empty area vs. 5 persons present and moving about.

Note: 
One of the other requirements of the project is to collect RSSI data from a minimum number of 3 access points.

## Method

	1. Data Collection

The first step of the project was to collect WiFi RSSI data at a specific location.

Since the assigned area had certain restrictions and problems like poor coverage and availability of university WiFi networks, our own WiFi APs had to be placed on fixed locations to collect data. A 4G home broadband router, a hotspot created using a mobile phone, and a 4G mobile WiFi router were used as WiFi APs.

WiFi signal strengths were obtained by a NodeMCU (ESP8266) device which was placed on a fixed location. The NodeMCU device was responsible for measuring RSSI of each WiFi AP and send them as a MQTT message to eclipse MQTT broker. The sent messages were obtained by a python script on a PC. The messages were in JSON format and the script converted them into Comma Separated Value (CSV) file on the PC. However, all the messages published by the NodeMCU could not be saved by the Python script due to the slow rate of receiving messages. Therefore, some of the originally published messages were discarded without being utilized, and due to that, a lot of time needed to be spent for collecting data.

The data collection step was done for all the 3 scenarios (no person, one person and 5 persons), and to obtain a good data set under different conditions, the same data collection procedure was carried out in 4 different days.

	2. Create the Data Set
	
After collecting data of all the 3 scenarios in 3 separate CSV files, they were combined to create one CSV file. A separate column “No_of_P” which indicates the type of scenario was added (Refer Figure 2.3) as a label to all the examples in the data set as follows:

* 0 - Indicates “No Person” scenario
* 1 - Indicates “One Person” scenario
* 5 - Indicates “Five Person” scenario

Then the data set was pre-processed to remove all the well seen outliers. During that process, several examples which were found with -100 dB signal strength to the continuously available access points were removed from the data set. This process was carried out only with the access points that have been substituted by us for the project. University available WiFi access points’ data were not processed using this method.

After removing the outliers, the data set was shuffled randomly to mix all the examples. Finally, a portion of created data set was saved as an unknown data set for the purpose of checking the prediction accuracy of the trained model.

	3. Select Suitable Classification Models
Once after the final data set is created, it was used to train a suitable model. In this project, six classification models which comes under the supervised learning method was initially compared with their accuracies for the above provided data set. Those models are:

 * Logistic Regression
 * Gaussian Naive Bayes
 * K Nearest Neighbour (K-NN)
 * Support Vector Machine (SVM)
 * Decision Tree
 * Extreme Gradient Boosting (XGBoost)
 
