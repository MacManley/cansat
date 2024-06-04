# Created by Toby Tangney

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from geopy.distance import geodesic

# Read the file while handling decoding errors
with open('/Users/Toby/Desktop/CanSatData/Input/SampledataT.TXT', 'rb') as file:
    try:
        data = file.readlines()
    except UnicodeDecodeError as e:
        print(f"Error decoding line: {e}")
        data = []

# Parse each JSON object
parsed_data = []
for line in data:
    line = line.decode('utf-8', errors='ignore').strip()
    # Attempt to replace common problematic control characters
    # For example, replace newline characters within the string
    line = line.replace('\n', '\\n')
    if line.startswith('{'):
        try:
            parsed_data.append(json.loads(line))
        except json.decoder.JSONDecodeError as e:
            print(f"Error parsing line: {e}")
            continue

# Convert the list of dictionaries into a Pandas DataFrame
df = pd.DataFrame(parsed_data)

# Create a dictionary for column mapping
column_mapping = {
    '0': 'Time',
    '1': 'AccelerationX',
    '2': 'AccelerationY',
    '3': 'AccelerationZ',
    '4': 'RotationX',
    '5': 'RotationY',
    '6': 'RotationZ',
    '7': 'TemperatureMPU',
    '8': 'Pressure',
    '9': 'Humidity',
    '10': 'Gas',
    '11': 'BME_Temperature',
    '12': 'UV_Voltage',
    '13': 'UV_Index',
    '14': 'Longitude',
    '15': 'Latitude',
    '16': 'datatime',
    '17': 'speed',
    '18': 'AltitudeGPS',
    '19': 'AltitudeBME'
}

# Rename columns using the mapping dictionary
df = df.rename(columns=column_mapping)

print(df.columns)

# Calculate the absolute acceleration
df['AccelerationAbsolute'] = np.sqrt(df['AccelerationX']**2 + df['AccelerationY']**2 + df['AccelerationZ']**2)
# Calculate the absolute gyroscope reading
df['GyroscopeAbsolute'] = np.sqrt(df['RotationX']**2 + df['RotationY']**2 + df['RotationZ']**2)

# Function to calculate speed
def calculate_speed(df):
    speeds = []
    for i in range(1, len(df)):
        # Calculate distance in meters between consecutive points
        distance = geodesic((df['Latitude'].iloc[i-1], df['Longitude'].iloc[i-1]),
                            (df['Latitude'].iloc[i], df['Longitude'].iloc[i])).meters
        # Convert time elapsed from milliseconds to seconds
        time_elapsed = (df['Time'].iloc[i] - df['Time'].iloc[i-1]) / 1000  # Convert ms to s
        # Calculate speed (m/s)
        speed = distance / time_elapsed if time_elapsed != 0 else 0
        speeds.append(speed)
    # Insert NaN for the first row as there's no speed value for it
    speeds.insert(0, float('nan'))
    return speeds

# Calculate speeds and add them as a new column to the DataFrame
# df['Speed'] = calculate_speed(df)
















#Output data from this ------

# Save the DataFrame as a CSV file
df.to_csv('/Users/Toby/Desktop/CanSatData/Output/TotalCSVFile/test3424.csv', index=False)
print("CSV file saved successfully.")


















#PRIAMARY MISSION GRAPHS - TEMPBME, TEMPU, PRESSUR, ALTITIDUEGPS, ALTITUDEBME TIME


# Temp V TEMP
plt.figure(figsize=(10, 6))
plt.plot(df['Time'], df['TemperatureMPU'] - 5, label='TemperatureMPU', marker='o', linestyle='-', color='r')
plt.plot(df['Time'], df['BME_Temperature'] - 5, label='BME_Temperature', marker='o', linestyle='-', color='b')

plt.title('Temp_MPU V Temp_BME')
plt.xlabel('Time')
plt.ylabel('Degrees Celcius')
plt.grid(True)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
# Save the figure as a PDF file
plt.savefig('/Users/Toby/Desktop/CanSatData/Output/Primary graphs/Indifividueal_Temp_Pressure_Altitude/TempBME V TempMPU.pdf', format='pdf')
# Clear the figure for the next plot
plt.close()
plt.clf()

# Pressure
plt.figure(figsize=(10, 6))
plt.plot(df['Time'], df['Pressure'], label='Pressure', marker='o', linestyle='-', color='g')

plt.title('Air Pressure')
plt.xlabel('Time')
plt.ylabel('HPA')
plt.grid(True)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
# Save the figure as a PDF file
plt.savefig('/Users/Toby/Desktop/CanSatData/Output/Primary graphs/Indifividueal_Temp_Pressure_Altitude/Pressure.pdf', format='pdf')
# Clear the figure for the next plot
plt.close()
plt.clf()


# AltitudeGPS
plt.figure(figsize=(10, 6))
plt.plot(df['Time'], df['AltitudeGPS'], label='AltitudeGPS', marker='o', linestyle='-', color='r')

plt.title('AltitudeGPS')
plt.xlabel('Time')
plt.ylabel('Meters above see level')
plt.grid(True)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
# Save the figure as a PDF file
plt.savefig('/Users/Toby/Desktop/CanSatData/Output/Primary graphs/Indifividueal_Temp_Pressure_Altitude/altitude.pdf', format='pdf')
# Clear the figure for the next plot
plt.close()
plt.clf()



# Temp V Pressure
plt.figure(figsize=(10, 6))

# Plot temperature and pressure
plt.plot(df['Time'], df['TemperatureMPU'], label='TemperatureMPU', marker='o', linestyle='-', color='r')
plt.plot(df['Time'], df['BME_Temperature'], label='BME_Temperature', marker='o', linestyle='-', color='b')
plt.plot(df['Time'], df['Pressure'] - 968, label='Pressure', marker='o', linestyle='-', color='g')

plt.title('Temp V Pressure')
plt.xlabel('Time')
plt.ylabel('num')
plt.grid(True)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()

# Save the figure as a PDF file
plt.savefig('/Users/Toby/Desktop/CanSatData/Output/Primary graphs/MixedCorilations/Temp V Pressure.pdf', format='pdf')

# Clear the figure for the next plot
plt.close()
plt.clf()







#DYDX graphs

#making code for equiation of the line and calculite derivitive of it to prove linear line
# Fit a linear regression model (trend line)
coefficients = np.polyfit(df['Time'], df['Pressure'], 1)
# coefficients[0] is the slope (m), coefficients[1] is the intercept (b)
slope = coefficients[0]
intercept = coefficients[1]
# Print the equation of the line
print(f"The equation of the trend line is: y = {slope:.2f}x + {intercept:.2f}")

# Equation of the line as a string
equation_str = f"y = {slope:.2f}x + {intercept:.2f}"

# Generate x and y values for the trend line
x_values = np.array(df['Time'])
y_values = slope * x_values + intercept

# Plotting the original data
plt.plot(df['Time'], df['Pressure'], color='blue', label='Pressure')

# Plotting the trend line
plt.plot(x_values, y_values, color='red', label='Linear Trend Line')

plt.text(x_values.mean(), y_values.min(), equation_str, fontsize=12, color='red')

# Adding labels and legend
plt.xlabel('Time')
plt.ylabel('Pressure')
plt.title('Pressure with Linear Trend Line')
plt.legend()
# Display the plot
plt.savefig('/Users/Toby/Desktop/CanSatData/Output/Primary graphs/DYDX/Trendequation.pdf', format='pdf')
plt.close()
plt.clf()



# Assuming coefficients are obtained from np.polyfit as before
# coefficients[0] is the slope (m), coefficients[1] is the intercept (b)
slope = coefficients[0]

# Generate x values across the same range as your original data
x_values = np.array(df['Time'])

# Since dy/dx is constant (slope), generate y values as a constant array of the slope value
y_values_derivative = np.full_like(x_values, slope)

# Plotting dy/dx
plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values_derivative, label='dy/dx', color='red')


# Adding titles and labels
plt.title('Derivative of Trend Line (dy/dx)')
plt.xlabel('Time')
plt.ylabel('dy/dx (Constant Value)')
plt.legend()

# Show grid for better readability
plt.grid(False)

plt.savefig('/Users/Toby/Desktop/CanSatData/Output/Primary graphs/DYDX/DYDX.pdf', format='pdf')
plt.close()
plt.clf()





# Altitude frome BME Pressure here

import matplotlib.pyplot as plt
import numpy as np

def altitude_from_pressure(pressure, pressure_sea_level=994):
    """
    Calculate altitude based on air pressure using the barometric formula.
    
    Args:
        pressure (float or array-like): The atmospheric pressure in hPa (hectopascals) or mbar (millibars).
        pressure_sea_level (float): Standard atmospheric pressure at sea level in hPa. 
                                    Default is 1013.25 hPa.
    
    Returns:
        float or array-like: Altitude in meters.
    """
    return ((-1) * np.log(pressure / pressure_sea_level) * 8.314 * 288.15) / (9.80665 * 0.0289644)

# Assuming df['Pressure'] contains pressure data
altitude = altitude_from_pressure(df['Pressure'])

plt.figure(figsize=(10, 6))
plt.plot(df['Time'], altitude, label='Altitude (from Pressure)', marker='o', linestyle='-', color='r')

plt.title('Altitude from Pressure')
plt.xlabel('Time')
plt.ylabel('Meters above sea level')
plt.grid(True)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()

# Save the figure as a PDF file
plt.savefig('/Users/Toby/Desktop/CanSatData/Output/Primary graphs/altitude_from_pressure.pdf', format='pdf')

# Clear the figure for the next plot
plt.close()
plt.clf()


























#Secondary graphs -----

# Now, plotting AccelerationAbsolute over Time
plt.figure(figsize=(10, 6))
plt.plot(df['Time'], df['AccelerationAbsolute'], label='Acceleration Absolute', marker='o', linestyle='-', color='r')
plt.title('Absolute Acceleration over Time')
plt.xlabel('Time')
plt.ylabel('Acceleration Absolute')
plt.grid(True)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
# Save the figure as a PDF file
plt.savefig('/Users/Toby/Desktop/CanSatData/Output/Secondary graphs/AbsoluteAcceleration_over_Time.pdf', format='pdf')
# Clear the figure for the next plot
plt.close()
plt.clf()


#Plotting graphs
plt.figure(figsize=(10, 6))  # Set the figure size (optional)
plt.plot(df['Time'], df['AccelerationX'], marker='o', linestyle='-', color='b')  # Line graph
plt.title('AccelerationX over Time')  # Title of the graph
plt.xlabel('Time')  # X-axis label
plt.ylabel('AccelerationX')  # Y-axis label
plt.grid(False)  # Show grid
plt.xticks(rotation=45)  # Rotate X-axis labels for better readability (optional)
plt.tight_layout()  # Automatically adjust subplot parameters to give specified padding

# Save the figure as a PDF file
plt.savefig('/Users/Toby/Desktop/CanSatData/Output/Secondary graphs/AccelerationX_over_Time.pdf', format='pdf')
plt.close()
plt.clf()



# Start a new figure
plt.figure(figsize=(12, 8))
# Sensors to plot
sensors = ['UV_Index']
colors = ['r', 'b', 'g', 'y']  # Colors for the plots and trend lines
markers = ['o', 'x', '^', 's']  # Markers for each sensor
linestyles = ['-', '--', '-.', ':']  # Line styles for each sensor

for sensor, color, marker, linestyle in zip(sensors, colors, markers, linestyles):
    # Plot sensor data
    plt.plot(df['Time'], df[sensor], label=sensor, marker=marker, linestyle=linestyle, color=color)
    
    # Calculate and plot trend line
    z = np.polyfit(df['Time'], df[sensor], 1)  # Fit a 1st degree polynomial (linear) to the data
    p = np.poly1d(z)  # Create polynomial function
    plt.plot(df['Time'], p(df['Time']), linestyle='-', color=color, linewidth=2, alpha=0.7, label=f'{sensor} Trend')

# Adding titles and labels
plt.title('Sensor Readings over Time with Trend Lines')
plt.xlabel('Time')
plt.ylabel('Sensor Values')

# Add a legend to differentiate the lines and trend lines
plt.legend()

# Show grid for better readability
plt.grid(False)

# Rotate X-axis labels for better readability
plt.xticks(rotation=45)

# Adjust layout
plt.tight_layout()

# Save the figure as a PDF file
plt.savefig('/Users/Toby/Desktop/CanSatData/Output/Secondary graphs/SensorReadingsTrends_over_Time.pdf', format='pdf')

print("Graph with trend lines saved as PDF successfully.")
plt.close()
plt.clf()








import matplotlib.pyplot as plt
import numpy as np

# Filter dataframe to remove rows with zero values
df_filtered = df[df['UV_Index'] > 0.4]

# Start a new figure
plt.figure(figsize=(12, 8))
# Sensors to plot
sensors = ['UV_Index']
colors = ['r', 'b', 'g', 'y']  # Colors for the plots and trend lines
markers = ['o', 'x', '^', 's']  # Markers for each sensor
linestyles = ['-', '--', '-.', ':']  # Line styles for each sensor

for sensor, color, marker, linestyle in zip(sensors, colors, markers, linestyles):
    # Plot sensor data
    plt.plot(df_filtered['Time'], df_filtered[sensor], label=sensor, marker=marker, linestyle=linestyle, color=color)
    
    # Calculate and plot trend line
    z = np.polyfit(df_filtered['Time'], df_filtered[sensor], 1)  # Fit a 1st degree polynomial (linear) to the data
    p = np.poly1d(z)  # Create polynomial function
    plt.plot(df_filtered['Time'], p(df_filtered['Time']), linestyle='-', color=color, linewidth=2, alpha=0.7, label=f'{sensor} Trend')

# Adding titles and labels
plt.title('Sensor Readings over Time with Trend Lines (Ignoring Zero Values)')
plt.xlabel('Time')
plt.ylabel('Sensor Values')

# Add a legend to differentiate the lines and trend lines
plt.legend()

# Show grid for better readability
plt.grid(False)

# Rotate X-axis labels for better readability
plt.xticks(rotation=45)

# Adjust layout
plt.tight_layout()

# Save the figure as a PDF file
plt.savefig('/Users/Toby/Desktop/CanSatData/Output/Secondary graphs/UVIndex.pdf', format='pdf')

print("Graph with trend lines (ignoring zero values) saved as PDF successfully.")
plt.close()
plt.clf()











 
# # Filter out zero speed values
# speed_nonzero = df[df['Speed'] != 0]['Speed']

# # Plotting speed
# plt.figure(figsize=(10, 6))
# plt.plot(speed_nonzero, label='Speed', marker='o', linestyle='-', color='r')
# plt.title('Speed')
# plt.xlabel('Time')
# plt.ylabel('Speed')
# plt.grid(True)
# plt.xticks(rotation=45)

# # Adjust the x and y axis limits
# plt.xlim(speed_nonzero.index.min(), speed_nonzero.index.max())  # Adjust x-axis limits
# plt.ylim(speed_nonzero.min() * 0.9, speed_nonzero.max() * 1.1)  # Adjust y-axis limits

# plt.legend()
# plt.tight_layout()

# # Save the figure as a PDF file
# plt.savefig('/Users/Toby/Desktop/CanSatData/Output/Secondary graphs/Speed.pdf', format='pdf')

# # Clear the figure for the next plot
# plt.close()
# plt.clf()








































# UNSUPERVISED MACHINE LEARNING
#DO KMEANS ON Epoch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt


# If you want to add the cluster labels back to the original DataFrame
df['Cluster'] = pd.NA  # Initialize the column with NA
df.loc[df.index, 'Cluster'] = df['Cluster']  # Assign cluster labels to the original DataFrame


# Verify the 'Cluster' column is created
if 'Cluster' in df.columns:
    print("'Cluster' column created successfully.")
else:
    print("Failed to create 'Cluster' column.")

# Normalize the data
scaler = StandardScaler()
numeric_df_time = df.select_dtypes(include=[np.number])
numeric_df_time.replace([np.inf, -np.inf], np.nan, inplace=True)
numeric_df_time.fillna(numeric_df_time.mean(), inplace=True)
data_normalized = scaler.fit_transform(numeric_df_time)

# Continue with the PCA and clustering code
pca = PCA(n_components=0.95)
data_pca = pca.fit_transform(data_normalized)

# Apply clustering on the PCA-reduced data
kmeans = KMeans(n_clusters=3, n_init=10, random_state=0)  
clusters = kmeans.fit_predict(data_pca)  # Correct usage: calling fit_predict on the kmeans instance

df['Cluster'] = clusters

# Add the PCA components to the dataframe for visualization
for i in range(data_pca.shape[1]):
    df[f'PCA_Component_{i}'] = data_pca[:, i]


for cluster in df['Cluster'].unique():
    clusterstats = df[df['Cluster'] == cluster].describe()

    clusterstats.to_csv('/Users/Toby/Desktop/CanSatData/Output/UnsupervisedML/STATSTOTAL.csv')



# Impute the missing values using the mean of each column
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputed_data = imputer.fit_transform(df)

# Now you can perform PCA on the imputed data
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(imputed_data)

# Plotting for the total DataFrame
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=df['Cluster'], cmap='viridis')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('PCA - Reduced Data (Total)')
plt.colorbar(label='Cluster')


image_filename = f'/Users/Toby/Desktop/CanSatData/Output/UnsupervisedML/cluster_plot_TOTAL.pdf'  # or '.pdf' for PDF file
plt.savefig(image_filename, bbox_inches='tight')  # bbox_inches='tight' is optional but often useful
plt.close()  # Close the figure to free up memory


df.to_csv('/Users/Toby/Desktop/CanSatData/Output/UnsupervisedML/csvfile.csv', index=False)




# Perform KMeans clustering on the imputed data
kmeans = KMeans(n_clusters=3, n_init=10, random_state=0)  # n_init=10 to match current behavior
df_cluster = kmeans.fit_predict(imputed_data)

# Perform PCA on the imputed data
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(imputed_data)

# Plotting for each user's DataFrame
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=df_cluster, cmap='viridis')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title(f'PCA - Reduced Data')
plt.colorbar(label='Cluster')

image_filename = f'/Users/Toby/Desktop/CanSatData/Output/UnsupervisedML/Output.pdf'
plt.savefig(image_filename, bbox_inches='tight')  # bbox_inches='tight' is optional but often useful
plt.close()  # Close the figure to free up memory

# Save the user DataFrame with the Cluster column
df_cluster = pd.DataFrame(df_cluster, columns=['Cluster']) # Convert NumPy array to a pandas DataFrame
df_cluster.to_csv('/Users/Toby/Desktop/CanSatData/Output/UnsupervisedML/Output2.pdf.csv', index=False)

clusterstats.to_csv(f'/Users/Toby/Desktop/CanSatData/Output/UnsupervisedML/Outputstats.pdf')














import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# Select variables for clustering and PCA
selected_variables = ['TemperatureMPU', 'Pressure', 'Humidity', 'Gas', 'BME_Temperature', 'UV_Index']  # Replace with the actual variable names

# Filter DataFrame to include only selected variables
selected_df = df[selected_variables].copy()

# Normalize the data
scaler = StandardScaler()
selected_df.replace([np.inf, -np.inf], np.nan, inplace=True)
selected_df.fillna(selected_df.mean(), inplace=True)
data_normalized = scaler.fit_transform(selected_df)

# Perform PCA on the selected variables
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_normalized)

# Perform KMeans clustering on the PCA-reduced data
kmeans = KMeans(n_clusters=3, n_init=10, random_state=0)
clusters = kmeans.fit_predict(data_pca)

# Add cluster labels to the DataFrame
selected_df['Cluster'] = clusters

# Plotting
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=clusters, cmap='viridis')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('PCA - Reduced Data')
plt.colorbar(label='Cluster')

# Save or show the plot
plt.savefig('/Users/Toby/Desktop/CanSatData/Output/UnsupervisedML/Picked_PCA_Clustering_Plot.pdf', bbox_inches='tight')


# Save the DataFrame with cluster labels
selected_df.to_csv('/Users/Toby/Desktop/CanSatData/Output/UnsupervisedML/PickedClustered_Data.csv', index=False)


















# Generatiang html maps for locaiton gps data

import folium
from folium.plugins import HeatMap
import os

save = "/Users/Toby/Desktop/CanSatData/Output/UnsupervisedML/Map"
os.makedirs(save, exist_ok=True)

# Create a Folium map centered around the mean latitude and longitude
map_center = [df['Latitude'].mean(), df['Longitude'].mean()]
mymap = folium.Map(location=map_center, zoom_start=10)

# Iterate through each row in your DataFrame and add a marker for each point
for index, row in df.iterrows():
    folium.Marker([row['Longitude'], row['Latitude']]).add_to(mymap)

# Save the map to an HTML file
mymap.save('/Users/Toby/Desktop/CanSatData/Output/UnsupervisedML/Map/Mapheatmap.html')

print("map done")



print("DONE")









#HEat maps


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Define function to generate correlation matrix and heatmap
def generate_correlation_heatmap(data, title, save_path):
    correlation_matrix = data.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title(title)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

# Select variables for clustering and PCA
selected_variables = ['TemperatureMPU', 'Pressure', 'Humidity', 'Gas', 'BME_Temperature']  # Replace with the actual variable names

# Filter DataFrame to include only selected variables
selected_df = df[selected_variables].copy()

title = "Heat Map"
generate_correlation_heatmap(selected_df, title, save_path="/Users/Toby/Desktop/CanSatData/Output/UnsupervisedML/CorrelationMatrix.pdf")

print("Ready")
