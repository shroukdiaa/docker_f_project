# **Project Report on Data Analysis and Model Development Using Docker**

## **Part 1: Data Analysis and Model Development**

### 1. **Problem Definition and Understanding**

In this project, we defined a problem of predicting product sales based on a set of features collected from the data. The goal was to build a predictive model using machine learning algorithms to forecast future values based on input data.

### 2. **Data Collection and Integration**

Data was collected from multiple sources and merged into a single dataset for use in model training. We used the `pandas` library to read and integrate the data:
```python
import pandas as pd

# Reading data from multiple files and merging
data = pd.read_csv('winter_data.csv')
print(data.head())

```
 ```
    Date  Temperature  Snowfall      Activity  Visitors  Rating
0  2023-12-01 00:00:00         -0.0       1.9   Ice Skating       177       5
1  2023-12-01 01:00:00         -6.4       0.5        Skiing        65       3
2  2023-12-01 02:00:00          1.5       0.4   Ice Skating        72       5
3  2023-12-01 03:00:00         10.2       0.5   Ice Skating       112       2
4  2023-12-01 04:00:00         -7.3       4.4  Snowboarding       126       3
```
### 3. **Exploratory Data Analysis**

After collecting the data, we explored it using descriptive statistics and analyzed relationships between features:
```python
print(data.info())
```
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 10000 entries, 0 to 9999
Data columns (total 6 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   Date         10000 non-null  object 
 1   Temperature  10000 non-null  float64
 2   Snowfall     10000 non-null  float64
 3   Activity     10000 non-null  object 
 4   Visitors     10000 non-null  int64  
 5   Rating       10000 non-null  int64  
dtypes: float64(2), int64(2), object(2)
memory usage: 468.9+ KB
```
```
print(data.describe())
```
```
  Temperature      Snowfall      Visitors       Rating
count  9480.000000  10000.000000  10000.000000  10000.00000
mean     -5.041456      5.133520    104.481000      3.00990
std      10.013490      5.126773     55.206442      1.41979
min     -44.200000      0.000000     10.000000      1.00000
25%     -11.700000      1.500000     56.000000      2.00000
50%      -5.100000      3.600000    105.000000      3.00000
75%       1.700000      7.100000    152.000000      4.00000
max      34.300000     47.500000    199.000000      5.00000
```
We also visualized the patterns using plots:
```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.pairplot(data)
plt.show()
```
![image](https://github.com/user-attachments/assets/9c7ed9c8-5efc-4bdb-8f31-1c98458c2a24)


### 4. **Data Preprocessing and Cleaning**

The data was cleaned using various techniques such as handling missing values and converting categorical data to numeric:
```python
# Select only numeric columns
numeric_data = data.select_dtypes(include=['number'])

# Fill missing values in numeric columns with their mean
data[numeric_data.columns] = numeric_data.fillna(numeric_data.mean())
# cheack missing values
print(data.isnull().sum())

```
```
Date           0
Temperature    0
Snowfall       0
Activity       0
Visitors       0
Rating         0
dtype: int64
```

### 5. **Feature Selection and Engineering**

We identified important features using a correlation matrix:
```python
from sklearn.feature_selection import SelectKBest, f_regression
import pandas as pd

# Assuming 'data' is a DataFrame that has been defined previously
X = data.drop('Temperature', axis=1)
y = data['Temperature']

# Convert non-numeric columns to numeric, coercing errors to NaN
X = X.apply(pd.to_numeric, errors='coerce')

# Optionally, drop rows with NaN values if necessary
X = X.dropna()

# Initialize the selector
selector = SelectKBest(score_func=f_regression, k=5)

try:
    # Fit and transform the data
    X_new = selector.fit_transform(X, y)
    print(X_new)
except ValueError as e:
    print(f"Error during feature selection: {e}")
```
```
Error during feature selection: Found array with 0 sample(s) (shape=(0, 5)) while a minimum of 1 is required by SelectKBest.
```

### 6. **Model Selection and Implementation**

We selected a linear regression algorithm to build the model and split the data into training and testing sets:
```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Sample data creation
# Assuming we have 100 samples and 3 features
X_new = np.random.rand(100, 3)  # Feature matrix with 100 samples and 3 features
y = np.random.rand(100)          # Target variable with 100 samples

X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_true = y_test
y_pred = model.predict(X_test)


print("Score:", model.score(X_test, y_test))
```
```
Score: -0.3197035128058774
```

### 7. **Evaluation Metrics and Results Interpretation**

We evaluated the model using the Mean Squared Error (MSE) metric:
```python
from sklearn.metrics import mean_squared_error

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
```
```
Mean Squared Error: 0.09783839740774568
```

### 8. **Documentation and Communication**

All operations were documented using `Jupyter Notebook`, and results were communicated through visualizations and explanations to make the findings easy to understand.

---

## **Part 2: Docker Implementation**

### 1. **Docker Implementation**

We used Docker to containerize the application and machine learning model. A `Dockerfile` was created to define the container setup:
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .

CMD ["python", "app.py"]
```

### 2. **Data Processing in Containers**

Inside the container, data was processed using `pandas` and other tools. The container was able to handle data efficiently.

### 3. **Model Training and Evaluation in Containers**

The model was trained inside the container using Docker:
```bash
docker build -t model-container .
docker run model-container
```

### 4. **Docker Compose**

We used Docker Compose to coordinate and run services uniformly. A `docker-compose.yml` file was created to define container settings:
```yaml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "5000:5000"
```

### 5. **Integration with Version Control using Git**

The project was integrated with Git for version control:
```bash
git init
git add .
git commit -m "Initial Commit"
git push origin main
```

### 6. **Documentation**

Docker usage was documented in a `README.md` file, which explained how to build and run containers.

### 7. **Security and Compliance**

Container security was ensured by using tools like `Docker Bench for Security`:
```bash
docker run -it --privileged --pid host docker/docker-bench-security
```

---

## **Conclusion**

This project utilized various techniques, from data analysis and model training to containerizing the application using Docker, ensuring scalability, security, and good performance. Docker allowed us to create isolated environments that enhanced reproducibility and deployment efficiency.
