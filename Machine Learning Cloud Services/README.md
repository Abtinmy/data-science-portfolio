# Overview
In this project, 4 different machine learning services are implemented. The input and output format of the services' API is JSON and its documantation can be found [here](static/swagger.json). There are also some input examples for each service in [here](input_outputs/).

![](images/icon.png)

Tech: Python, Pandas, Scikit-learn, Docker, Flask, Matplotlib, Imblearn, statsmodes, khayyam

# Services
   * Service 1: Time Series Interpolation Service
   
     This Service takes some Date-Time data (either in Chiristian(miladi) or Persian(shamsi) date-time format) along with some values. The service then useses interpolation methods to resample missing data. Performing linear, polynomial, and spline interpolation methods on daily, monthly, hour, minute, and seconds time series data. For miladi format I used the pre-defined function in pandas and for shamsi I convert it to miladi and then performed the interpolation and again convert it to shamsi format. make sure to provide an appropriate order in request's config parameter when using spline or polynomial method.

   * Service 2: Time Series Interpolation and Date Format Transformation from Christian(miladi) to Persian(shamsi) Service
   
     This service is almost the same as the first service. The only difference is that in this service, input dates are always in miladi format and the output dates are all in shamsi format.
     
   * Service 3: Outlier Detection Service
   
     This service takes some data which can be time-series or not, and then it uses different outlier detection methods to predict whether a data is outlier or not. The output is a JSON consisting of 5 (if the data format is not time-series) or 3 (if the data is time-series) parts. One part is the id or the date-time and other parts are the results of applying different interpolation methods on the dataset. For instance, if the second element of the first part of the output(excluding the id) is false then the second element is not considered to be an outlier by the first outlier detection method. In the response message, indexes with true values are outliers.
     The methods used for the anomaly detection are listed as follows:
     - isolation forest
     - local outlier factor
     - IQR
     - Hampel method
     - z-score based method
     - Minimum Covariance Determinant
     - One class SVM
     - seasonl trend decomposition
     - seasonl ESD

   * Service 4: Imbalanced Data Management Service
   
     This service takes a dataset including the class of each data point(which can be 0 or 1). Then it uses oversampling, undersampling or SMOTE methods to generate a new balanced dataset and then it returns the new dataset as the output.
     performed methods are listed as follows:
     - Near miss
     - Random over sampler
     - SMOTE
     - Random under sampling
     - Borderline SMOTE
     - SVMSMOTE
     - ADASYN


# Installation
The Code is written in Python 3.9. If you don't have Python installed you can find it [here](https://www.python.org/downloads/). If you are using a lower version of Python you can upgrade using the pip package, ensuring you have the latest version of pip. To install the required packages and libraries, run this command in the project directory after [cloning](https://www.howtogeek.com/451360/how-to-clone-a-github-repository/) the repository:
```bash
pip install -r requirements.txt
```

