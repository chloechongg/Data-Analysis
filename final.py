# DATA UNDERSTANDING 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


#load dataset
df = pd.read_csv("C:\\Users\\chloe\\Downloads\\Life Expectancy Data.csv")
#display first few rows
print(df.head())

#data set overview
print(df.info())

#check for missing values
print(df.isnull().sum())

# CATEGORICAL VARIABLES
# print categorical variables
categorical_columns = df.select_dtypes(include=['object']).columns
print("Categorical Variables:")
print(categorical_columns)

# Replace missing values in categorical columns with the mode
categorical_columns = ['Country', 'Status']
for col in categorical_columns:
    df[col] = df[col].fillna(df[col].mode()[0])

print("Remaining Missing Values:")
print(df.isnull().sum())

binary_variables = []
non_binary_variables = []

for col in categorical_columns:
    unique_values = df[col].nunique()  # Count unique values
    if unique_values == 2:
        binary_variables.append(col)
    else:
        non_binary_variables.append(col)

print("\nBinary Variables:")
print(binary_variables)

print("\nNon-Binary Variables:")
print(non_binary_variables)


print("\nFrequency Counts for Each Categorical Variable:")
for col in categorical_columns:
    print(f"\n{col}:")
    print(df[col].value_counts())


print("\nFrequency Distribution for Each Categorical Variable:")
for col in categorical_columns:
    print(f"\n{col}:")
    print(df[col].value_counts(normalize=True))

print("\nCardinality (Number of Labels) for Each Categorical Variable:")
for col in categorical_columns:
    num_labels = df[col].nunique()  # Get the number of unique labels (categories)
    print(f"{col}: {num_labels} unique labels")


print("\nLabels for Each Categorical Variable:")
for col in categorical_columns:
    print(f"\n{col}:")
    print(df[col].unique())  # Print the unique labels in the column


print("\nOne-Hot Encoding for Each Categorical Variable (k-1 dummy variables):")
for col in categorical_columns:
    # Perform One-Hot Encoding, dropping the first category to avoid the dummy variable trap
    one_hot_encoded = pd.get_dummies(df[col], drop_first=True)
    
    # Preview the first few rows
    print(f"\nOne-Hot Encoded Variables for {col}:")
    print(one_hot_encoded.head())

print("\nSummary Statistics for Categorical Variables:")
print(df[categorical_columns].describe())

# NUMERICAL VALUES

# print numerical variables
numerical_columns = df.select_dtypes(include=['number']).columns
print("Numerical Variables:")
print(numerical_columns)

# Replace missing values in numerical columns
numerical_columns = [
    'Life expectancy ', 'Adult Mortality', 'Alcohol', 'percentage expenditure',
    'Hepatitis B', ' BMI ', 'Polio', 'Total expenditure', 'Diphtheria ', 'GDP', 
    'Population', ' thinness  1-19 years', ' thinness 5-9 years',
    'Income composition of resources', 'Schooling'
]

for col in numerical_columns:
    if col in ['Adult Mortality', 'GDP', 'Population']:  # Columns likely to have outliers
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna(df[col].mean())

print("Remaining Missing Values:")
print(df.isnull().sum())

continuous_variables = []
discontinuous_variables = []

for col in numerical_columns:
    unique_values = df[col].nunique()
    if unique_values > 10:  # Arbitrary threshold, typically continuous variables have many unique values
        continuous_variables.append(col)
    else:
        discontinuous_variables.append(col)

print("\nContinuous Variables:")
print(continuous_variables)

print("\nDiscontinuous Variables:")
print(discontinuous_variables)

print("\nFrequency Counts for Each Numerical Variable:")
for col in numerical_columns:
    print(f"\n{col}:")
    print(df[col].value_counts())

print("\nFrequency Distribution for Each Numerical Variable:")
for col in numerical_columns:
    print(f"\n{col}:")
    print(df[col].value_counts(normalize=True))


print("\nCardinality (Number of Labels) for Each Numerical Variable:")
for col in numerical_columns:
    num_labels = df[col].nunique()  # Get the number of unique labels (categories)
    print(f"{col}: {num_labels} unique labels")

print("\nLabels for Each Numerical Variable:")
for col in numerical_columns:
    print(f"\n{col}:")
    print(df[col].unique())  # Print the unique labels in the column

print("\nOne-Hot Encoding for Each Numerical Variable (k-1 dummy variables):")
for col in numerical_columns:
    # Perform One-Hot Encoding, dropping the first category to avoid the dummy variable trap
    one_hot_encoded = pd.get_dummies(df[col], drop_first=True)
    
    # Preview the first few rows
    print(f"\nOne-Hot Encoded Variables for {col}:")
    print(one_hot_encoded.head())

print("\nSummary Statistics for Numerical Variables:")
print(df[numerical_columns].describe())

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(15, 10))

for i, col in enumerate(numerical_columns, 1):
    plt.subplot(len(numerical_columns)//3 + 1, 3, i)  # Adjust the subplot grid based on the number of columns
    sns.boxplot(x=df[col])
    plt.title(f'Box Plot of {col}')
    
plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 10))

for i, col in enumerate(numerical_columns, 1):
    plt.subplot(len(numerical_columns)//3 + 1, 3, i)
    sns.histplot(df[col], kde=True)  # Kernel Density Estimation (KDE) for smoother visualization
    plt.title(f'Histogram of {col}')

plt.tight_layout()
plt.show()

from scipy import stats

for col in continuous_variables:
    # Check if the distribution is approximately normal using skewness
    skewness = df[col].skew()
    if abs(skewness) < 0.5:  # Assume normal if skewness is near 0
        # Extreme value analysis: Z-score method
        z_scores = stats.zscore(df[col].dropna())
        outliers = df[col][(z_scores > 3) | (z_scores < -3)]  # Outliers beyond 3 standard deviations
        print(f"\nExtreme values in {col} (normal distribution):")
        print(outliers)
    else:
        print(f"\n{col} is not normally distributed, skipping extreme value analysis.")


# Step 7: Calculate IQR for Skewed Variables and Print Outlier Statements
for col in continuous_variables:
    skewness = df[col].skew()
    if abs(skewness) > 0.5:  # If the distribution is skewed
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Print outlier values with a statement like "outlier values < lower_bound or > upper_bound"
        outliers = df[col][(df[col] < lower_bound) | (df[col] > upper_bound)]
        
        if not outliers.empty:
            print(f"\nOutlier values for {col}:")
            print(f"Outlier values < {lower_bound} or > {upper_bound}")
        else:
            print(f"\nNo outliers in {col} based on IQR analysis.")

# Correlation Analysis
# Exclude non-numeric columns
numeric_data = df.select_dtypes(include=[np.number])

# Calculate correlation matrix
correlation_matrix = numeric_data.corr()

print("The correlation matrix shows the relationship between different numeric variables in the dataset. Values close to +1 indicate a strong positive correlation, while values close to -1 indicate a strong negative correlation. Values around 0 suggest little to no linear relationship. This matrix helps to identify variables that move in the same or opposite direction, which could inform further analysis or feature selection.")

# Plot the correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# Exclude non-numeric columns
numeric_data = df.select_dtypes(include=[np.number])

# Calculate covariance matrix
covariance_matrix = numeric_data.cov()

# Display the covariance matrix
print("\nCovariance Matrix:")
print(covariance_matrix)

print("""
The covariance matrix represents the relationship between pairs of variables in terms of their joint variability. 
A positive covariance indicates that as one variable increases, the other tends to increase, while a negative covariance indicates that as one variable increases, the other tends to decrease. 
The magnitude of covariance is influenced by the scale of the variables, so it may not be directly comparable between different variable pairs. 
This matrix is useful for understanding the strength and direction of relationships between variables in their original units.
""")

# Define a custom function to format the annotations
def format_annotations(val):
    return f"{val:.2f}" if abs(val) < 1000 else f"{val:.1e}"  # Use scientific notation for large values

# Visualize covariance matrix with increased cell size and reduced text size
plt.figure(figsize=(20, 16))  # Increase the figure size
sns.heatmap(
    covariance_matrix, 
    annot=True, 
    fmt=".2f",  # Format to two decimal places
    annot_kws={"size": 7},  # Reduce font size
    linewidths=0.5,         # Add spacing between cells
    linecolor="gray",       # Add lines for better separation
    cmap="Blues",           # Use a different colormap
    cbar_kws={'label': 'Covariance'},  # Label for color bar
    square=True  # Ensure square-shaped cells
)

# Apply the custom annotation formatting
for text in plt.gca().texts:  # Modify each annotation text
    text.set_text(format_annotations(float(text.get_text())))

plt.title("Covariance Matrix", fontsize=16)
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()  # Ensure no clipping of labels
plt.show()

# DATA PREPARATION

import webbrowser
import os

# Open the file in the default browser
file_path = "C:\\Users\\chloe\\OneDrive - University of Mary\\fall 2024\\CSC 370 01\\Life Expectancy Data.htm"
webbrowser.open('file://' + os.path.realpath(file_path))

# stratified sampling for sampling technique

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

# Load your dataset
df = pd.read_csv("C:\\Users\\chloe\\Downloads\\Life Expectancy Data.csv")

# Ensure that 'GDP' has no NaN values
df['GDP'] = df['GDP'].fillna(df['GDP'].mean())  # You can fill with the mean, median, or other strategy

# Create bins for the 'GDP' column (You can adjust the bin edges as needed)
bins = [0, 1000, 10000, 50000, 100000]  # Define GDP ranges
labels = ['Low', 'Medium', 'High', 'Very High']  # Labels for these ranges
df['GDP_category'] = pd.cut(df['GDP'], bins=bins, labels=labels, right=False)

# Check the distribution of the new categories in the 'GDP_category' column
print(df['GDP_category'].value_counts())

# Add 'Unknown' as a category to the 'GDP_category' column, if necessary
df['GDP_category'] = df['GDP_category'].cat.add_categories('Unknown')

# Ensure there are no NaN values in 'GDP_category' after binning (if there are any, fill them with 'Unknown')
df['GDP_category'] = df['GDP_category'].fillna('Unknown')

# Initialize StratifiedShuffleSplit (you can adjust test_size for the split ratio)
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

# Apply the stratified sampling to your dataset
for train_idx, test_idx in splitter.split(df, df['GDP_category']):
    train_set = df.iloc[train_idx]
    test_set = df.iloc[test_idx]

# You now have the stratified train and test datasets
print("\nStratified Train Set:")
print(train_set.head())

print("Stratified Test Set:")
print(test_set.head())

# Explanation of the analysis
print("\nExplanation:")
print("In this analysis, we applied stratified sampling to the dataset based on the 'GDP' column. "
      "This ensures that each stratum (i.e., different GDP levels such as Low, Medium, High, and Very High) "
      "is proportionally represented in both the training and testing datasets. "
      "By doing this, we account for variations in income and ensure that our models are trained on data "
      "that accurately reflects the distribution of different GDP levels. "
      "The stratified sampling technique is especially useful when analyzing large and diverse datasets, "
      "as it helps prevent underrepresentation of certain groups (in this case, GDP categories) "
      "and makes the results more robust and generalizable.")

# multiple regression analysis for statistical test

# Load your dataset
df = pd.read_csv("C:\\Users\\chloe\\Downloads\\Life Expectancy Data.csv")

# Define the dependent variable (column name)
dependent_var = 'Life expectancy '

# Get the independent variables by dropping the dependent variable
independent_vars = df.drop(columns=[dependent_var]).select_dtypes(include=['number']).columns

# Remove rows with missing values in the dependent and independent variables
df_clean = df.dropna(subset=[dependent_var] + list(independent_vars))

# Define independent variables (X) and dependent variable (y)
X = df_clean[independent_vars]  # Independent variables
y = df_clean[dependent_var]  # Dependent variable

# Ensure y is numeric
if not np.issubdtype(y.dtype, np.number):
    raise ValueError(f"The dependent variable '{dependent_var}' contains non-numeric data.")

# Fit the multiple regression model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

model = LinearRegression()
model.fit(X, y)

# Evaluate the model performance
y_pred = model.predict(X)

# Print the model coefficients
print("\nModel Coefficients (effect of each variable on Life Expectancy):")
for var, coef in zip(independent_vars, model.coef_):
    print(f"{var}: {coef:.4f}")

print("""
Interpretation of coefficients:
The coefficients indicate the expected change in life expectancy for a one-unit increase in the corresponding variable, 
while holding all other variables constant. A positive coefficient suggests a positive relationship with life expectancy, 
whereas a negative coefficient indicates a negative relationship.
""")

# Print the intercept
print("Intercept (baseline Life Expectancy when all variables are zero): ")
print(f"Intercept: {model.intercept_:.4f}")

print("""
Interpretation of intercept:
The intercept represents the baseline predicted value of life expectancy when all independent variables are zero. 
This value may not always have a meaningful real-world interpretation, depending on the context of the dataset.
""")

# Print evaluation metrics
mse = mean_squared_error(y, y_pred)
r_squared = r2_score(y, y_pred)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print("""
Interpretation of MSE:
MSE measures the average squared difference between the observed and predicted values. 
Lower MSE values indicate better model performance. However, it is influenced by the scale of the dependent variable.
""")

# Calculate RMSE
rmse = sqrt(mean_squared_error(y, y_pred))

# Print the RMSE result
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print("\nInterpretation of RMSE: \nRMSE represents the average prediction error in the original units of the dependent variable (Life Expectancy).")
print(f"In this case, an RMSE of {rmse:.4f} years means that the model's predictions are off by approximately {rmse:.2f} years, on average, from the actual life expectancy values.\n")


print(f"R-squared: {r_squared:.4f}")
print(f"""
Interpretation of R-squared:
R-squared indicates the proportion of variance in life expectancy explained by the independent variables. 
An R-squared of {r_squared:.4f} means that approximately {r_squared * 100:.2f}% of the variability in life expectancy 
is accounted for by the model. Higher values suggest a better fit, but overly high values may indicate overfitting.
""")

print("""Summary of Results
Key Variables with Strongest Impacts:

Income composition of resources (10.4701): A strong positive coefficient suggests a substantial increase in life expectancy with higher income composition, highlighting the importance of socio-economic development.
Schooling (0.9063): Indicates a positive association between education and life expectancy, reinforcing the value of education in health outcomes.
HIV/AIDS (-0.4495): A significant negative impact, meaning higher prevalence of HIV/AIDS drastically reduces life expectancy.
Other Notable Variables:

Adult Mortality (-0.0164): Higher adult mortality rates slightly reduce life expectancy.
BMI (0.0316): Suggests a minor positive effect, as healthier BMI levels contribute to longer lives.
Alcohol (-0.0983): Indicates that increased alcohol consumption is linked to lower life expectancy, though the impact is relatively small.
GDP and Population (near 0): Minimal or no influence, potentially due to scaling issues or collinearity with other socio-economic factors.
Model Evaluation:

MSE (12.5377): The model's average squared error; smaller MSE reflects reasonable prediction accuracy.
RMSE (3.5409): Predicts life expectancy with an average error of ~3.54 years, a decent result given the data's complexity.
R-squared (0.8379): The model explains ~83.8% of the variance in life expectancy, indicating a good fit without severe overfitting.
Intercept (313.3528):

The baseline life expectancy when all variables are zero. While it may not have practical meaning (since zeroing variables like schooling and income is unrealistic), it anchors the regression model.""")

# MODELING 

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv("C:\\Users\\chloe\\Downloads\\Life Expectancy Data.csv")

# Define the dependent variable (target) and independent variables (features)
dependent_var = 'Life expectancy '  # Dependent variable (target)
independent_vars = df.drop(columns=['Country', 'Year', 'Status', dependent_var]).select_dtypes(include=['number']).columns

# Remove rows with missing values
df_clean = df.dropna(subset=[dependent_var] + list(independent_vars))

# Define features (X) and target (y)
X = df_clean[independent_vars]  # Independent variables
y = df_clean[dependent_var]  # Dependent variable

# Feature scaling (Standardizing the data)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------ 1. Linear Regression ------------------------
# Initialize and train the Linear Regression model
linear_model = LinearRegression()

# Perform K-Fold Cross Validation for Linear Regression
kf = KFold(n_splits=10, shuffle=True, random_state=42)
linear_reg_cv_scores = cross_val_score(linear_model, X_scaled, y, cv=kf, scoring='neg_mean_squared_error')

# Print results for Linear Regression
mean_mse = -linear_reg_cv_scores.mean()  # MSE is returned as negative by cross_val_score
std_mse = linear_reg_cv_scores.std()

print("\nLinear Regression - K-Fold Cross Validation Results (MSE):")
print(f"Mean MSE: {mean_mse:.4f}")
print(f"Standard Deviation of MSE: {std_mse:.4f}")

# ------------------------ 2. Logistic Regression ------------------------
# For Logistic Regression, we'll convert the target variable to categorical.
# Create a binary classification by classifying Life Expectancy as high or low (median split).
median_life_expectancy = df_clean[dependent_var].median()
y_binary = np.where(y >= median_life_expectancy, 1, 0)  # 1 if Life Expectancy is higher than median, 0 otherwise

# Initialize and train the Logistic Regression model
logistic_model = LogisticRegression(max_iter=1000)

# Perform K-Fold Cross Validation for Logistic Regression
logistic_reg_cv_scores = cross_val_score(logistic_model, X_scaled, y_binary, cv=kf, scoring='accuracy')

# Print results for Logistic Regression
mean_accuracy = logistic_reg_cv_scores.mean()
std_accuracy = logistic_reg_cv_scores.std()

print("\nLogistic Regression - K-Fold Cross Validation Results (Accuracy):")
print(f"Mean Accuracy: {mean_accuracy:.4f}")
print(f"Standard Deviation of Accuracy: {std_accuracy:.4f}")

# Train and evaluate Logistic Regression on the entire dataset (optional)
logistic_model.fit(X_scaled, y_binary)
y_pred_binary = logistic_model.predict(X_scaled)
accuracy = accuracy_score(y_binary, y_pred_binary)
conf_matrix = confusion_matrix(y_binary, y_pred_binary)

print("\nLogistic Regression - Accuracy on Full Dataset:")
print(f"Accuracy: {accuracy:.4f}")
print("Confusion Matrix:")
print(conf_matrix)

# Output interpretation
print("\n--- Linear Regression Results ---")
print("Linear Regression - K-Fold Cross Validation Results (MSE):")
print(f"Mean MSE: {mean_mse:.4f}")
print(f"Standard Deviation of MSE: {std_mse:.4f}")
print("\nExplanation:")
print(f"The Mean MSE of {mean_mse:.4f} indicates the average squared difference between the actual and predicted life expectancy values.")
print("A lower MSE would indicate better prediction accuracy, but since MSE is influenced by the scale of life expectancy, this value is typical for the scale of your data.")
print(f"The standard deviation of MSE ({std_mse:.4f}) shows that the MSE scores across different folds of the cross-validation have a relatively small spread, suggesting stable model performance.")

print("\n--- Logistic Regression Results ---")
print("Logistic Regression - K-Fold Cross Validation Results (Accuracy):")
print(f"Mean Accuracy: {mean_accuracy:.4f}")
print(f"Standard Deviation of Accuracy: {std_accuracy:.4f}")
print("\nExplanation:")
print(f"The mean accuracy of {mean_accuracy:.4f} (or {mean_accuracy * 100:.2f}%) means that, on average, the logistic regression model correctly predicted life expectancy categories (high vs. low) for {mean_accuracy * 100:.2f}% of the cases during cross-validation.")
print("This suggests that the model is performing quite well in distinguishing between high and low life expectancy regions.")
print(f"The standard deviation of accuracy ({std_accuracy:.4f}) is low, meaning that the model's accuracy is relatively consistent across the different folds of the cross-validation.")

print("\n--- Logistic Regression Evaluation on Full Dataset ---")
print("Logistic Regression - Accuracy on Full Dataset:")
print(f"Accuracy: {accuracy:.4f}")
print("\nExplanation:")
print(f"The accuracy of {accuracy:.4f} (or {accuracy * 100:.2f}%) indicates that when trained on the entire dataset, the logistic regression model correctly predicted the life expectancy category for {accuracy * 100:.2f}% of the data.")
print("This result is consistent with the cross-validation accuracy, suggesting that the model generalizes well to the full dataset.")

print("\nConfusion Matrix:")
print(conf_matrix)
print("\nExplanation:")
print("The confusion matrix provides insights into the model's performance:")
print(f"True Positives (TP): {conf_matrix[1, 1]} - The number of high life expectancy regions correctly predicted.")
print(f"True Negatives (TN): {conf_matrix[0, 0]} - The number of low life expectancy regions correctly predicted.")
print(f"False Positives (FP): {conf_matrix[0, 1]} - The number of low life expectancy regions incorrectly predicted as high.")
print(f"False Negatives (FN): {conf_matrix[1, 0]} - The number of high life expectancy regions incorrectly predicted as low.\n")

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Initialize and train the Random Forest Regressor model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_scaled, y)

# Predict using Random Forest Regressor model
y_pred_rf = rf_model.predict(X_scaled)

# Compute Mean Squared Error (MSE) for Random Forest Regressor
mse_rf = mean_squared_error(y, y_pred_rf)

# Print the MSE for Random Forest Regressor
print("Random Forest Regressor - Mean Squared Error (MSE):")
print(f"MSE: {mse_rf:.4f}\n")

print("""Since the MSE (0.4359) of the Random Forest Regressor is significantly lower than the MSE 
(13.1767) of the Linear Regression Model, this indicates a better performance and reduced losses
from the Random Forest Regressor.\n""")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Convert Life Expectancy to a binary classification (high vs low based on median)
median_life_expectancy = df_clean[dependent_var].median()
y_binary = (y >= median_life_expectancy).astype(int)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_binary, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Classifier model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict the classes on the test set
y_pred_rf = rf_model.predict(X_test)

# Compute F1 score for Random Forest Classifier
f1_rf = f1_score(y_test, y_pred_rf)

# Compute accuracy for Random Forest Classifier
accuracy_rf = accuracy_score(y_test, y_pred_rf)

# Print F1 score and accuracy
print(f"Random Forest Classifier - F1 Score: {f1_rf:.4f}")
print(f"Random Forest Classifier - Accuracy: {accuracy_rf:.4f}\n")

print("""Since the accuracy value(0.9394) from the Random Forest Classifier (RFC) is greater than
the accuracy value from the Logistic Regression Model(0.8666), this indicates the RFC is better 
at classifying the test data based on the features provided""")

