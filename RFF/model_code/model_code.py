import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import pickle

# Load the dataset
df = pd.read_csv(r"C:\Users\zain2\Downloads\Compressed\House_Rent_Dataset.csv")

df.pop("Posted On")
df.pop("Floor")
df.pop("Area Locality")
df.pop("Point of Contact")
df.pop("Area Type")
df.pop("Tenant Preferred")

df['City'] = df['City'].replace(["Mumbai", "Bangalore", "Hyderabad", "Delhi", "Chennai", "Kolkata"], [5, 4, 3, 2, 1, 0])
df['Furnishing Status'] = df['Furnishing Status'].replace(["Furnished", "Semi-Furnished", "Unfurnished"], [2, 1, 0])

target = df.pop("Rent")

x_train, x_test, y_train, y_test = train_test_split(df, target, test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.fit(x_train, y_train)

train_accuracy = model.score(x_train, y_train)
test_accuracy = model.score(x_test, y_test)

print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)

# Save the trained model to a file
with open('model/regression_model.pkl', 'wb') as file:
    pickle.dump(model, file)
