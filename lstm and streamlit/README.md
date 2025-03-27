So what we plan to achieve with this application is it being able to closely predict the temperature for next day for New Delhi. Now, the question is how will be able to do this ? Lets answer that. 

Lets first source the dataset for the project. For this project we are going to use the weather dataset available on Kaggle. The dataset contains features such as Date, Pressure, Temperature, Windspeed, Humidity. We are going to create a model that is going to predict temperature using past 7 days data. Since the data is a sequential data we are going to use a LSTM model to train our model which will be able to capture the sequential relationship in the data and can maximize the predictions. 


