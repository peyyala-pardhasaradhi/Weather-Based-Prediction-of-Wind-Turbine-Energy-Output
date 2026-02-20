import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load the cleaned data from Step 1
df = pd.read_csv(r'C:\Users\surya\Desktop\Wind_Power_Project\data\cleaned_wind_data.csv')

# Define Features and Target
X = df[['Speed', 'Theoretical', 'Direction']]
y = df['Power']

# Build and Train the Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Save the model into your project folder
model_path = r'C:\Users\surya\Desktop\Wind_Power_Project\power_prediction.sav'
pickle.dump(model, open(model_path, 'wb'))
print("Success: power_prediction.sav created!")