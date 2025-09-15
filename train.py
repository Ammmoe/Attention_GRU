# --- Test trajectory_loader ---
from data.trajectory_loader import load_and_concat_flights

csv_path = "data/flights.csv"  # replace with your actual CSV path
min_rows = 1000
num_flights = 5

try:
    # Load and concatenate flights
    data_for_model = load_and_concat_flights(
        csv_path, min_rows=min_rows, num_flights=num_flights
    )

    # Check the output
    print("Data shape:", data_for_model.shape)
    print("Column names:", data_for_model.columns.tolist())
    print("First 5 rows:\n", data_for_model.head(10))

except Exception as e:
    print("Error loading flights:", e)
