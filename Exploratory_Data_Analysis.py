import pandas as pd
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
# Load your dataset
df = pd.read_csv("enhanced_ui_preference_dataset.csv")


df_encoded = df.copy()

label_encoders = {}
for column in df_encoded.columns:
    if df_encoded[column].dtype == 'object':
        le = LabelEncoder()
        df_encoded[column] = le.fit_transform(df_encoded[column])
        label_encoders[column] = le


plt.figure(figsize=(14, 10))
corr = df_encoded.corr()

# Focus only on correlation with Feedback and Learning Style
target_corr = corr[["Feedback", "Learning Style"]].sort_values(by="Feedback", ascending=False)

sns.heatmap(target_corr, annot=True, cmap="coolwarm")
plt.title("Correlation with Feedback and Learning Style")
plt.show()


# Prepare features and target
X = df_encoded.drop(columns=["Feedback", "User ID", "Learning Style"])
y = df_encoded["Feedback"]  # For now just focus on Feedback

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Feature importance
importances = model.feature_importances_
feature_names = X.columns

# Sort and plot
sorted_indices = importances.argsort()[::-1]
plt.figure(figsize=(12, 6))
sns.barplot(x=importances[sorted_indices], y=feature_names[sorted_indices])
plt.title("Feature Importance for Predicting User Feedback")
plt.show()
# Group by Learning Style and look at mean values
df.groupby("Learning Style")[["Clicks", "Time Spent (s)", "Scroll Depth (%)"]].mean().plot(kind="bar", figsize=(12, 6))
plt.title("Behavioral Patterns per Learning Style")
plt.ylabel("Mean Values")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()



