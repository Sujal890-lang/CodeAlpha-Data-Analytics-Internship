# ============================================================
# CodeAlpha Data Analytics Internship - Task 3: Data Visualization
# Author: Sujal
# Description: Advanced Data Visualizations on Amazon Products
#              Dataset - Creating compelling visual stories
# ============================================================

# ── STEP 0: Install required libraries ──────────────────────
# pip install pandas numpy matplotlib seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  CodeAlpha Internship | Task 3: Data Visualization")
print("=" * 60)


# ── STEP 1: Load Dataset ─────────────────────────────────────

CSV_PATH = "amazon_cleaned_dataset.csv"

try:
    df = pd.read_csv(CSV_PATH)
    print(f"\n✅ Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
except FileNotFoundError:
    print(f"\n❌ '{CSV_PATH}' not found. Run Task 1 first!")
    exit()


# ── STEP 2: Set Visual Style ──────────────────────────────────

plt.style.use('seaborn-v0_8-darkgrid')
COLORS = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12',
          '#9b59b6', '#1abc9c', '#e67e22', '#34495e']


# ── VISUALIZATION 1: Top 10 Categories (Horizontal Bar) ──────

if 'Category' in df.columns:
    fig, ax = plt.subplots(figsize=(12, 7))
    cat_counts = df['Category'].value_counts().head(10)
    bars = ax.barh(cat_counts.index, cat_counts.values, color=COLORS)

    # Add value labels on bars
    for bar, val in zip(bars, cat_counts.values):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                str(val), va='center', fontsize=10, fontweight='bold')

    ax.set_title('🛒 Top 10 Product Categories on Amazon',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Number of Products', fontsize=12)
    ax.set_ylabel('Category', fontsize=12)
    plt.tight_layout()
    plt.savefig('viz1_top_categories.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Visualization 1 saved: viz1_top_categories.png")


# ── VISUALIZATION 2: Rating Distribution (KDE + Histogram) ───

if 'Rating' in df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(df['Rating'].dropna(), bins=20,
                 color='#3498db', edgecolor='white', alpha=0.8)
    axes[0].set_title('Rating Histogram', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Rating')
    axes[0].set_ylabel('Count')
    axes[0].axvline(df['Rating'].mean(), color='red',
                    linestyle='--', label=f"Mean: {df['Rating'].mean():.2f}")
    axes[0].legend()

    # KDE Plot
    sns.kdeplot(df['Rating'].dropna(), ax=axes[1],
                fill=True, color='#2ecc71', alpha=0.6)
    axes[1].set_title('Rating Density (KDE)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Rating')
    axes[1].set_ylabel('Density')

    fig.suptitle('⭐ Product Rating Analysis', fontsize=16,
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('viz2_rating_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Visualization 2 saved: viz2_rating_analysis.png")


# ── VISUALIZATION 3: Discount % by Category (Box Plot) ───────

if 'Category' in df.columns and 'Discount_Percent' in df.columns:
    top5_cats = df['Category'].value_counts().head(5).index
    df_top5 = df[df['Category'].isin(top5_cats)]

    fig, ax = plt.subplots(figsize=(13, 6))
    sns.boxplot(data=df_top5, x='Category', y='Discount_Percent',
                palette=COLORS[:5], ax=ax)
    ax.set_title('💰 Discount % Distribution by Top 5 Categories',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Category', fontsize=12)
    ax.set_ylabel('Discount Percentage (%)', fontsize=12)
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    plt.savefig('viz3_discount_boxplot.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Visualization 3 saved: viz3_discount_boxplot.png")


# ── VISUALIZATION 4: Price Savings Heatmap ───────────────────

if 'Category' in df.columns and 'Price_Savings' in df.columns and 'Rating' in df.columns:
    top5_cats = df['Category'].value_counts().head(5).index
    df_heat = df[df['Category'].isin(top5_cats)]

    pivot = df_heat.pivot_table(values='Price_Savings',
                                index='Category',
                                columns='Rating_Label',
                                aggfunc='mean') if 'Rating_Label' in df.columns else None

    if pivot is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd',
                    linewidths=0.5, ax=ax)
        ax.set_title('🔥 Average Price Savings by Category & Rating',
                     fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig('viz4_heatmap.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("✅ Visualization 4 saved: viz4_heatmap.png")


# ── VISUALIZATION 5: Scatter Plot - Price vs Rating ──────────

if 'Actual_Price' in df.columns and 'Rating' in df.columns:
    fig, ax = plt.subplots(figsize=(12, 6))

    scatter = ax.scatter(
        df['Actual_Price'].dropna(),
        df['Rating'].dropna(),
        alpha=0.4,
        c=df['Discount_Percent'] if 'Discount_Percent' in df.columns else '#3498db',
        cmap='coolwarm',
        s=30
    )

    if 'Discount_Percent' in df.columns:
        plt.colorbar(scatter, label='Discount %')

    ax.set_title('💎 Product Price vs Rating (colored by Discount %)',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Actual Price (₹)', fontsize=12)
    ax.set_ylabel('Rating', fontsize=12)
    ax.set_xlim(0, df['Actual_Price'].quantile(0.95))
    plt.tight_layout()
    plt.savefig('viz5_price_vs_rating.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Visualization 5 saved: viz5_price_vs_rating.png")


# ── VISUALIZATION 6: Dashboard Summary ───────────────────────

fig = plt.figure(figsize=(16, 10))
fig.suptitle('📊 Amazon Products - Data Story Dashboard',
             fontsize=20, fontweight='bold', y=0.98)

# Plot A: Category counts
ax1 = fig.add_subplot(2, 3, 1)
if 'Category' in df.columns:
    top5 = df['Category'].value_counts().head(5)
    ax1.pie(top5.values, labels=top5.index, autopct='%1.1f%%',
            colors=COLORS[:5], startangle=90)
    ax1.set_title('Top 5 Categories', fontweight='bold')

# Plot B: Rating histogram
ax2 = fig.add_subplot(2, 3, 2)
if 'Rating' in df.columns:
    ax2.hist(df['Rating'].dropna(), bins=15, color='#3498db', edgecolor='white')
    ax2.set_title('Rating Distribution', fontweight='bold')
    ax2.set_xlabel('Rating')

# Plot C: Discount histogram
ax3 = fig.add_subplot(2, 3, 3)
if 'Discount_Percent' in df.columns:
    ax3.hist(df['Discount_Percent'].dropna(), bins=15,
             color='#e74c3c', edgecolor='white')
    ax3.set_title('Discount % Distribution', fontweight='bold')
    ax3.set_xlabel('Discount %')

# Plot D: Rating label bar
ax4 = fig.add_subplot(2, 3, 4)
if 'Rating_Label' in df.columns:
    label_counts = df['Rating_Label'].value_counts()
    ax4.bar(label_counts.index, label_counts.values, color=COLORS[:4])
    ax4.set_title('Rating Labels', fontweight='bold')
    ax4.set_ylabel('Count')

# Plot E: Top categories avg rating
ax5 = fig.add_subplot(2, 3, 5)
if 'Category' in df.columns and 'Rating' in df.columns:
    avg_r = df.groupby('Category')['Rating'].mean().sort_values(ascending=False).head(5)
    ax5.barh(avg_r.index, avg_r.values, color=COLORS[:5])
    ax5.set_title('Avg Rating by Category', fontweight='bold')
    ax5.set_xlabel('Avg Rating')

# Plot F: Price savings bar
ax6 = fig.add_subplot(2, 3, 6)
if 'Category' in df.columns and 'Price_Savings' in df.columns:
    avg_savings = df.groupby('Category')['Price_Savings'].mean().sort_values(ascending=False).head(5)
    ax6.bar(range(len(avg_savings)), avg_savings.values, color=COLORS[:5])
    ax6.set_xticks(range(len(avg_savings)))
    ax6.set_xticklabels(avg_savings.index, rotation=20, ha='right', fontsize=8)
    ax6.set_title('Avg Price Savings by Category', fontweight='bold')
    ax6.set_ylabel('Savings (₹)')

plt.tight_layout()
plt.savefig('viz6_dashboard.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Visualization 6 saved: viz6_dashboard.png")


# ── FINAL SUMMARY ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("  KEY DATA STORY - AMAZON PRODUCTS")
print("=" * 60)
if 'Category' in df.columns:
    print(f"\n🏆 Top Category    : {df['Category'].value_counts().index[0]}")
if 'Rating' in df.columns:
    print(f"⭐ Avg Rating      : {df['Rating'].mean():.2f} / 5.0")
if 'Discount_Percent' in df.columns:
    print(f"💰 Avg Discount    : {df['Discount_Percent'].mean():.1f}%")
if 'Price_Savings' in df.columns:
    print(f"💵 Avg Savings     : ₹{df['Price_Savings'].mean():.0f}")
print(f"📦 Total Products  : {len(df)}")

print("\n📁 All 6 visualizations saved as PNG files!")
print("\n" + "=" * 60)
print("  ✅ Task 3 Complete!")
print("=" * 60)