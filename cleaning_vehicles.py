import pandas as pd
import numpy as np
from datetime import datetime

# Load the data
df = pd.read_csv(r"C:\Users\User\Desktop\IDS\vehicles.csv")

# ===== BASIC CLEANING =====
# 1. Handle missing values
categorical_cols = ['manufacturer', 'model', 'condition', 'cylinders', 
                   'fuel', 'title_status', 'transmission', 'drive', 
                   'size', 'type', 'paint_color', 'county']

for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].fillna('Unknown')

# 2. Numeric columns - fill with median
numeric_cols = ['price', 'year', 'odometer']
for col in numeric_cols:
    if col in df.columns:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

# 3. Drop columns with too many missing values or not useful
cols_to_drop = ['VIN', 'image_url', 'description', 'region_url']
df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')

# ===== DATA TYPE CONVERSION =====
# More robust datetime conversion
if 'posting_date' in df.columns:
    df['posting_date'] = pd.to_datetime(df['posting_date'], errors='coerce', utc=True)
    # Convert from UTC to naive datetime if needed
    df['posting_date'] = df['posting_date'].dt.tz_localize(None)

if 'year' in df.columns:
    df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(0).astype(int)

# ===== OUTLIER HANDLING =====
if 'price' in df.columns:
    df = df[(df['price'] > 500) & (df['price'] < 200000)]  # $500-$200k range

if 'year' in df.columns:
    current_year = datetime.now().year
    df = df[(df['year'] >= 1900) & (df['year'] <= current_year + 1)]

if 'odometer' in df.columns:
    df = df[(df['odometer'] >= 0) & (df['odometer'] <= 500000)]  # 0-500k miles

# ===== FEATURE ENGINEERING =====
if 'year' in df.columns:
    current_year = datetime.now().year
    df['age'] = current_year - df['year']

if all(col in df.columns for col in ['price', 'odometer']):
    df['price_per_mile'] = np.where(df['odometer'] > 0, 
                                  df['price'] / df['odometer'], 
                                  np.nan)

if 'posting_date' in df.columns:
    # Only extract features if datetime conversion was successful
    if pd.api.types.is_datetime64_any_dtype(df['posting_date']):
        df['posting_year'] = df['posting_date'].dt.year
        df['posting_month'] = df['posting_date'].dt.month
        df['posting_day_of_week'] = df['posting_date'].dt.dayofweek
    else:
        print("Warning: Could not extract datetime features from posting_date")

# ===== CATEGORY CLEANING =====
if 'condition' in df.columns:
    df['condition'] = df['condition'].astype(str).str.lower().str.strip()

if 'fuel' in df.columns:
    df['fuel'] = df['fuel'].astype(str).str.lower().str.strip()

if 'transmission' in df.columns:
    transmission_map = {
        'automatic': 'automatic',
        'manual': 'manual',
        'other': 'other',
        'unknown': 'unknown'
    }
    df['transmission'] = (df['transmission'].astype(str)
                          .str.lower()
                          .map(transmission_map)
                          .fillna('unknown'))

# ===== FINAL TOUCHES =====
df = df.reset_index(drop=True)

# Save cleaned data
df.to_csv(r'C:\Users\User\Desktop\IDS\cleaned_vehicles.csv', index=False)

print(f"Data cleaning complete. Original shape: {df.shape}, New shape: {df.shape}")
print("Columns with missing values after cleaning:")
print(df.isnull().sum()[df.isnull().sum() > 0])