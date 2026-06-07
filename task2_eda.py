# ============================================================
# CodeAlpha Data Analytics Internship - Task 2: EDA
# Author: Sujal
# Description: Exploratory Data Analysis on Amazon Products
#              Dataset cleaned in Task 1
# ============================================================

# ── STEP 0: Install required libraries ──────────────────────
# Run this in terminal before executing:
#   pip install pandas numpy matplotlib seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set plot style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 60)
print("  CodeAlpha Internship | Task 2: EDA")
print("=" * 60)


# ── STEP 1: Load Cleaned Dataset ─────────────────────────────

CSV_PATH = "amazon_cleaned_dataset.csv"

try:
    df = pd.read_csv(CSV_PATH)
    print(f"\n✅ Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
except FileNotFoundError:
    print(f"\n❌ ERROR: '{CSV_PATH}' not found.")
    print("   Please run task1_web_scraping.py first!")
    exit()


# ── STEP 2: Basic Info ────────────────────────────────────────

print("\n📋 COLUMN NAMES:")
print(df.columns.tolist())

print("\n📊 DATA TYPES:")
print(df.dtypes)

print("\n❓ MISSING VALUES:")
print(df.isnull().sum())

print("\n📈 BASIC STATISTICS:")
print(df.describe())


# ── STEP 3: Ask Meaningful Questions ─────────────────────────

print("\n" + "=" * 60)
print("  ANALYSIS QUESTIONS")
print("=" * 60)

# Q1: How many products per category?
print("\n🔹 Q1: How many products are in each category?")
if 'Category' in df.columns:
    cat_counts = df['Category'].value_counts().head(10)
    print(cat_counts)

# Q2: What is the average rating?
print("\n🔹 Q2: What is the average product rating?")
if 'Rating' in df.columns:
    print(f"   Average Rating: {df['Rating'].mean():.2f} / 5.0")
    print(f"   Highest Rating: {df['Rating'].max()}")
    print(f"   Lowest Rating:  {df['Rating'].min()}")

# Q3: Which category has best average rating?
print("\n🔹 Q3: Which category has the best average rating?")
if 'Category' in df.columns and 'Rating' in df.columns:
    best_cat = df.groupby('Category')['Rating'].mean().sort_values(ascending=False).head(5)
    print(best_cat)

# Q4: What is average discount?
print("\n🔹 Q4: What is the average discount percentage?")
if 'Discount_Percent' in df.columns:
    print(f"   Average Discount: {df['Discount_Percent'].mean():.2f}%")
    print(f"   Max Discount:     {df['Discount_Percent'].max()}%")

# Q5: Rating distribution
print("\n🔹 Q5: How are ratings distributed?")
if 'Rating_Label' in df.columns:
    print(df['Rating_Label'].value_counts())


# ── STEP 4: Visualizations ────────────────────────────────────

print("\n📊 Generating visualizations...")

# --- Plot 1: Top 10 Categories by Product Count ---
if 'Category' in df.columns:
    plt.figure(figsize=(12, 6))
    cat_counts = df['Category'].value_counts().head(10)
    sns.barplot(x=cat_counts.values, y=cat_counts.index, palette='Blues_r')
    plt.title('Top 10 Categories by Number of Products', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Products')
    plt.ylabel('Category')
    plt.tight_layout()
    plt.savefig('plot1_category_counts.png', dpi=150)
    plt.show()
    print("✅ Plot 1 saved: plot1_category_counts.png")

# --- Plot 2: Rating Distribution ---
if 'Rating' in df.columns:
    plt.figure(figsize=(10, 5))
    sns.histplot(df['Rating'].dropna(), bins=20, kde=True, color='steelblue')
    plt.title('Distribution of Product Ratings', fontsize=16, fontweight='bold')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('plot2_rating_distribution.png', dpi=150)
    plt.show()
    print("✅ Plot 2 saved: plot2_rating_distribution.png")

# --- Plot 3: Average Rating by Category ---
if 'Category' in df.columns and 'Rating' in df.columns:
    plt.figure(figsize=(12, 6))
    avg_rating = df.groupby('Category')['Rating'].mean().sort_values(ascending=False).head(10)
    sns.barplot(x=avg_rating.values, y=avg_rating.index, palette='Greens_r')
    plt.title('Top 10 Categories by Average Rating', fontsize=16, fontweight='bold')
    plt.xlabel('Average Rating')
    plt.ylabel('Category')
    plt.tight_layout()
    plt.savefig('plot3_avg_rating_category.png', dpi=150)
    plt.show()
    print("✅ Plot 3 saved: plot3_avg_rating_category.png")

# --- Plot 4: Discount % Distribution ---
if 'Discount_Percent' in df.columns:
    plt.figure(figsize=(10, 5))
    sns.histplot(df['Discount_Percent'].dropna(), bins=20, kde=True, color='coral')
    plt.title('Distribution of Discount Percentages', fontsize=16, fontweight='bold')
    plt.xlabel('Discount %')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('plot4_discount_distribution.png', dpi=150)
    plt.show()
    print("✅ Plot 4 saved: plot4_discount_distribution.png")

# --- Plot 5: Rating Label Pie Chart ---
if 'Rating_Label' in df.columns:
    plt.figure(figsize=(8, 8))
    label_counts = df['Rating_Label'].value_counts()
    colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
    plt.pie(label_counts.values, labels=label_counts.index,
            autopct='%1.1f%%', colors=colors, startangle=140)
    plt.title('Product Rating Labels Distribution', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plot5_rating_labels.png', dpi=150)
    plt.show()
    print("✅ Plot 5 saved: plot5_rating_labels.png")

# --- Plot 6: Actual Price vs Discounted Price ---
if 'Actual_Price' in df.columns and 'Discounted_Price' in df.columns:
    plt.figure(figsize=(10, 6))
    sample = df[['Actual_Price', 'Discounted_Price']].dropna().head(50)
    plt.scatter(sample['Actual_Price'], sample['Discounted_Price'],
                alpha=0.6, color='purple')
    plt.plot([0, sample['Actual_Price'].max()],
             [0, sample['Actual_Price'].max()],
             'r--', label='No Discount Line')
    plt.title('Actual Price vs Discounted Price', fontsize=16, fontweight='bold')
    plt.xlabel('Actual Price (₹)')
    plt.ylabel('Discounted Price (₹)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plot6_price_comparison.png', dpi=150)
    plt.show()
    print("✅ Plot 6 saved: plot6_price_comparison.png")


# ── STEP 5: Key Insights Summary ─────────────────────────────

print("\n" + "=" * 60)
print("  KEY INSIGHTS FROM EDA")
print("=" * 60)

if 'Category' in df.columns:
    top_cat = df['Category'].value_counts().index[0]
    print(f"\n🏆 Most Popular Category : {top_cat}")

if 'Rating' in df.columns:
    avg_r = df['Rating'].mean()
    print(f"⭐ Average Rating        : {avg_r:.2f} / 5.0")

if 'Discount_Percent' in df.columns:
    avg_d = df['Discount_Percent'].mean()
    print(f"💰 Average Discount      : {avg_d:.1f}%")

if 'Rating_Label' in df.columns:
    top_label = df['Rating_Label'].value_counts().index[0]
    print(f"📊 Most Common Rating    : {top_label}")

print(f"\n📦 Total Products Analyzed: {len(df)}")

print("\n" + "=" * 60)
print("  ✅ Task 2 Complete! All plots saved as PNG files.")
print("=" * 60)