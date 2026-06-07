# ============================================================
# CodeAlpha Data Analytics Internship - Task 1: Web Scraping
# Author: Sujal
# Description: Extract, clean and create a custom dataset
#              from Amazon Sales Dataset (Kaggle)
# ============================================================

# ── STEP 0: Install required libraries ──────────────────────
# Run this in terminal before executing:
#   pip install pandas requests beautifulsoup4 lxml

import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import os

print("=" * 60)
print("  CodeAlpha Internship | Task 1: Web Scraping")
print("=" * 60)


# ── STEP 1: Load the Kaggle Amazon Dataset ───────────────────
# Make sure amazon.csv is in the same folder as this script.

CSV_PATH = "amazon_data.csv"   # <-- Change path if needed

try:
    df = pd.read_csv(CSV_PATH)
    print(f"\n✅ Dataset loaded successfully!")
    print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
except FileNotFoundError:
    print(f"\n❌ ERROR: '{CSV_PATH}' not found.")
    print("   Please place amazon.csv in the same folder as this script.")
    exit()


# ── STEP 2: Explore Raw Data ─────────────────────────────────

print("\n📋 Column Names:")
print(df.columns.tolist())

print("\n📊 First 3 rows (raw):")
print(df.head(3).to_string())

print("\n🔍 Data Types:")
print(df.dtypes)

print("\n❓ Missing Values:")
print(df.isnull().sum())


# ── STEP 3: Select & Rename Relevant Columns ─────────────────
# Keep only the most useful columns for analysis

COLUMNS_NEEDED = {
    'product_id'        : 'Product_ID',
    'product_name'      : 'Product_Name',
    'category'          : 'Category',
    'discounted_price'  : 'Discounted_Price',
    'actual_price'      : 'Actual_Price',
    'discount_percentage': 'Discount_Percent',
    'rating'            : 'Rating',
    'rating_count'      : 'Rating_Count',
    'about_product'     : 'About_Product',
}

# Keep only columns that exist in the file
existing_cols = {k: v for k, v in COLUMNS_NEEDED.items() if k in df.columns}
df = df[list(existing_cols.keys())].rename(columns=existing_cols)

print(f"\n✅ Selected {len(existing_cols)} relevant columns.")


# ── STEP 4: Clean the Data ────────────────────────────────────

def clean_price(val):
    """Remove ₹ symbol and commas, convert to float."""
    if pd.isna(val):
        return None
    val = str(val).replace('₹', '').replace(',', '').strip()
    try:
        return float(val)
    except:
        return None

def clean_percent(val):
    """Remove % sign, convert to float."""
    if pd.isna(val):
        return None
    val = str(val).replace('%', '').strip()
    try:
        return float(val)
    except:
        return None

def clean_rating(val):
    """Convert rating to float, handle 'out of 5' style strings."""
    if pd.isna(val):
        return None
    val = str(val).split()[0]
    try:
        return float(val)
    except:
        return None

def clean_rating_count(val):
    """Remove commas from rating count."""
    if pd.isna(val):
        return None
    val = str(val).replace(',', '').strip()
    try:
        return int(val)
    except:
        return None

def clean_category(val):
    """Keep only the top-level category (before first |)."""
    if pd.isna(val):
        return None
    return str(val).split('|')[0].strip()

# Apply cleaning functions
if 'Discounted_Price' in df.columns:
    df['Discounted_Price'] = df['Discounted_Price'].apply(clean_price)

if 'Actual_Price' in df.columns:
    df['Actual_Price'] = df['Actual_Price'].apply(clean_price)

if 'Discount_Percent' in df.columns:
    df['Discount_Percent'] = df['Discount_Percent'].apply(clean_percent)

if 'Rating' in df.columns:
    df['Rating'] = df['Rating'].apply(clean_rating)

if 'Rating_Count' in df.columns:
    df['Rating_Count'] = df['Rating_Count'].apply(clean_rating_count)

if 'Category' in df.columns:
    df['Category'] = df['Category'].apply(clean_category)

print("\n✅ Data cleaned successfully!")


# ── STEP 5: Remove Duplicates & Handle Missing Values ────────

before = len(df)
df.drop_duplicates(subset=['Product_ID'], keep='first', inplace=True)
after = len(df)
print(f"\n🗑️  Removed {before - after} duplicate rows.")

# Drop rows where critical fields are missing
df.dropna(subset=['Product_Name', 'Rating'], inplace=True)
print(f"✅ Final dataset size: {len(df)} rows")


# ── STEP 6: Add Engineered Features ──────────────────────────

# Price savings
if 'Actual_Price' in df.columns and 'Discounted_Price' in df.columns:
    df['Price_Savings'] = df['Actual_Price'] - df['Discounted_Price']

# Rating category
def rate_label(r):
    if r >= 4.5: return 'Excellent'
    elif r >= 4.0: return 'Good'
    elif r >= 3.0: return 'Average'
    else: return 'Poor'

df['Rating_Label'] = df['Rating'].apply(rate_label)

print("\n✅ New features added: Price_Savings, Rating_Label")


# ── STEP 7: Summary Statistics ───────────────────────────────

print("\n📈 Summary Statistics:")
print(df.describe(include='all').to_string())

print("\n🏷️ Top 5 Categories by Product Count:")
if 'Category' in df.columns:
    print(df['Category'].value_counts().head())

print("\n⭐ Average Rating by Category:")
if 'Category' in df.columns and 'Rating' in df.columns:
    print(df.groupby('Category')['Rating'].mean().sort_values(ascending=False).head(10))


# ── STEP 8: Save Custom Dataset ──────────────────────────────

OUTPUT_FILE = "amazon_cleaned_dataset.csv"
df.to_csv(OUTPUT_FILE, index=False)

print(f"\n💾 Custom dataset saved as: '{OUTPUT_FILE}'")
print(f"   Final shape: {df.shape[0]} rows × {df.shape[1]} columns")

print("\n" + "=" * 60)
print("  ✅ Task 1 Complete! Dataset ready for EDA (Task 2).")
print("=" * 60)