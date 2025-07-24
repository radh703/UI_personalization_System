import pandas as pd
import random
from faker import Faker
import numpy as np
import logging
from datetime import datetime
import re
from scipy.stats import iqr


# Set up logging
logging.basicConfig(filename='data_cleaning_log.txt', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Starting data cleaning process")

df = pd.read_csv('dataset.csv')

# --- Comprehensive Data Cleaning ---
try:
    # 1. Log initial dataset statistics
    initial_rows = len(df)
    initial_nulls = df.isnull().sum().sum()
    logging.info(f"Initial dataset: {initial_rows} rows, {initial_nulls} missing values")

    # 2. Handle missing values with context-aware imputation
    df["Gender"] = df["Gender"].fillna("Unknown")
    df["Location"] = df["Location"].fillna("Unknown")
    df["Internet Speed"] = df["Internet Speed"].replace("", "Medium").fillna("Medium")
    df["Time of Day"] = df["Time of Day"].replace("", "Day").fillna("Day")
    df["UI Theme"] = df["UI Theme"].replace("", "System Default").fillna("System Default")
    df["Font Size"] = df.apply(
        lambda row: "Small" if row["Device Type"] in ["Mobile", "mobile"] and pd.isna(row["Font Size"]) 
        else "Medium" if pd.isna(row["Font Size"]) else row["Font Size"], axis=1)
    df["Feedback"] = df["Feedback"].replace("", "Unknown").fillna("Unknown")
    df["Layout Preference"] = df["Layout Preference"].replace("", "Grid").fillna("Grid")
    df["Navigation Style"] = df["Navigation Style"].replace("", "Top Menu").fillna("Top Menu")
    df["Color Palette"] = df["Color Palette"].replace("", "Blue Dominant").fillna("Blue Dominant")
    df["Interaction Level"] = df["Interaction Level"].replace("", "Minimal").fillna("Minimal")
    df["Component Density"] = df["Component Density"].replace("", "Standard").fillna("Standard")
    df["Button Style"] = df["Button Style"].replace("", "Rounded").fillna("Rounded")
    df["Icon Style"] = df["Icon Style"].replace("", "Filled").fillna("Filled")
    df["Loading Style"] = df["Loading Style"].replace("", "Spinner").fillna("Spinner")
    df["Error Style"] = df["Error Style"].replace("", "Text-only").fillna("Text-only")
    df["Animation Level"] = df["Animation Level"].replace("", "Subtle").fillna("Subtle")
    df["Learning Style"] = df["Learning Style"].replace("", "Watch videos").fillna("Watch videos")
    logging.info(f"Missing values handled: {df.isnull().sum().sum()} missing values remaining")

    # 3. Standardize categorical variables using mapping
    standardization_map = {
        "Device Type": {"mobile": "Mobile", "desktop": "Desktop", "tablet": "Tablet", "": "Unknown"},
        "OS": {"ios": "iOS", "windows": "Windows", "macos": "macOS", "linux": "Linux", "android": "Android", "unknown": "Unknown"},
        "Internet Speed": {"slow": "Slow", "medium": "Medium", "fast": "Fast", "": "Medium"},
        "Time of Day": {"day": "Day", "night": "Night", "": "Day"},
        "UI Theme": {"light": "Light", "dark": "Dark", "system default": "System Default", "": "System Default"},
        "Font Size": {"small": "Small", "medium": "Medium", "large": "Large", "": "Medium"},
        "Feedback": {"good": "Good", "bad": "Bad", "": "Unknown"},
        "Layout Preference": {"grid": "Grid", "list": "List", "card-based": "Card-based", "single-column": "Single-column", "": "Grid"},
        "Navigation Style": {"top menu": "Top Menu", "hamburger menu": "Hamburger Menu", "tab bar": "Tab Bar", "gesture-based": "Gesture-based"},
        "Color Palette": {"blue dominant": "Blue Dominant", "warm tones": "Warm Tones", "high contrast": "High Contrast", 
                         "pastel": "Pastel", "monochrome": "Monochrome", "": "Blue Dominant"},
        "Component Density": {"compact": "Compact", "standard": "Standard", "spacious": "Spacious", "": "Standard"},
        "Icon Style": {"filled": "Filled", "outlined": "Outlined", "duotone": "Duotone", "hand-drawn": "Hand-drawn", "": "Filled"},
        "Error Style": {"text only": "Text-only", "text-only": "Text-only", "illustration": "Illustration", "toast": "Toast", "modal": "Modal", "": "Text-only"}
    }
    for column, mapping in standardization_map.items():
        df[column] = df[column].str.lower().replace(mapping).str.title()
        logging.info(f"Standardized column: {column}")

    # 4. Clean string fields (remove special characters, extra spaces)
    def clean_string(s):
        if pd.isna(s):
            return s
        return re.sub(r'[^\w\s:-]', '', s.strip())
    
    df["Location"] = df["Location"].apply(clean_string)
    df["Search Query"] = df["Search Query"].apply(clean_string)
    df["Customization Choices"] = df["Customization Choices"].apply(clean_string)
    logging.info("String fields cleaned (Location, Search Query, Customization Choices)")

    # 5. Fix invalid numerical values
    df.loc[df["Age"] < 0, "Age"] = 18
    df.loc[df["Age"] > 100, "Age"] = 60
    df.loc[df["Clicks"] < 0, "Clicks"] = 1
    df.loc[df["Time Spent (s)"] < 0, "Time Spent (s)"] = 10
    df.loc[df["Scroll Depth (%)"] < 0, "Scroll Depth (%)"] = 0
    df.loc[df["Scroll Depth (%)"] > 100, "Scroll Depth (%)"] = 100
    df.loc[df["Session Duration (s)"] < 0, "Session Duration (s)"] = 100
    logging.info("Invalid numerical values corrected (Age, Clicks, Time Spent, Scroll Depth, Session Duration)")

    # 6. Handle outliers using IQR method
    def cap_outliers(series, multiplier=1.5):
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr_val = iqr(series)
        lower_bound = q1 - multiplier * iqr_val
        upper_bound = q3 + multiplier * iqr_val
        return series.clip(lower=lower_bound, upper=upper_bound)

    numerical_columns = ["Clicks", "Time Spent (s)", "Session Duration (s)"]
    for col in numerical_columns:
        df[col] = cap_outliers(df[col])
        logging.info(f"Outliers capped for {col} using IQR method")

    # 7. Validate logical consistency
    df.loc[(df["Animation Level"].isin(["Moderate", "Expressive"])) & (~df["Interaction Level"].isin(["Moderate", "High"])), 
           "Interaction Level"] = "Moderate"
    logging.info("Logical consistency enforced for Animation Level and Interaction Level")

    # 8. Handle duplicates
    duplicate_count = df.duplicated(subset=["User ID"]).sum()
    if duplicate_count > 0:
        df = df.groupby("User ID").agg({
            col: "first" if df[col].dtype == "object" else "mean" for col in df.columns
        }).reset_index()
        logging.info(f"Resolved {duplicate_count} duplicate User IDs by merging")

    # 9. Feature engineering: Create engagement_level
    df["Engagement Level"] = pd.qcut(df["Time Spent (s)"] + df["Clicks"] * 10, q=3, labels=["Low", "Medium", "High"])
    logging.info("Created new feature: Engagement Level")

    # 10. Data quality report
    final_rows = len(df)
    final_nulls = df.isnull().sum().sum()
    quality_report = f"""
    Data Cleaning Summary:
    - Initial Rows: {initial_rows}
    - Final Rows: {final_rows}
    - Rows Dropped: {initial_rows - final_rows}
    - Initial Missing Values: {initial_nulls}
    - Final Missing Values: {final_nulls}
    - Numerical Columns Cleaned: {numerical_columns}
    - Categorical Columns Standardized: {list(standardization_map.keys())}
    - New Features Added: ['Engagement Level']
    """
    logging.info(quality_report)
    with open("data_quality_report.txt", "w") as f:
        f.write(quality_report)

    # Save cleaned dataset
    df.to_csv("cleaned_ui_preference_dataset.csv", index=False)
    logging.info("Cleaned dataset saved as cleaned_ui_preference_dataset.csv")

except Exception as e:
    logging.error(f"Error during cleaning: {str(e)}")
    raise

print("Messy UI Preference Dataset Generated and Extensively Cleaned!")
print("Cleaning log saved to 'data_cleaning_log.txt'")
print("Data quality report saved to 'data_quality_report.txt'")
print(df[["Learning Style", "Engagement Level"]].head())