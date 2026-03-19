import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============ LOAD DATA ============
df = pd.read_csv('tested.csv')

print("=" * 50)
print("🤖 WEEK 3 — TITANIC DATA EXPLORER")
print("=" * 50)

# ============ BASIC EXPLORATION ============
print("\n📊 BASIC INFO:")
print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}")
df.info()

# ============ MISSING VALUES ============
print("\n📊 MISSING VALUES:")
print(df.isnull().sum())

# ============ FILL MISSING VALUES ============
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Fare'] = df['Fare'].fillna(df['Fare'].mean())
print("\n✅ Missing values filled!")

# ============ STATISTICS ============
print("\n📈 STATISTICS SUMMARY:")
print(df.describe())

# ============ FILTERING ============
print("\n🔍 FILTERING:")

# Survivors only
survivors = df[df['Survived'] == 1]
print(f"Total survivors: {len(survivors)}")

# First class only
first_class = df[df['Pclass'] == 1]
print(f"First class passengers: {len(first_class)}")

# Old survivors — age > 30 AND survived
old_survivors = df[(df['Age'] > 30) & (df['Survived'] == 1)]
print(f"Survivors older than 30: {len(old_survivors)}")

# ============ GROUPBY ANALYSIS ============
print("\n📊 SURVIVAL RATE BY CLASS:")
print(df.groupby('Pclass')['Survived'].mean() * 100)

print("\n📊 DETAILED CLASS ANALYSIS:")
print(df.groupby('Pclass')['Survived'].agg(['mean', 'count', 'sum']))

print("\n📊 AVERAGE AGE BY CLASS:")
print(df.groupby('Pclass')['Age'].mean())

# ============ KEY INSIGHTS ============
print("\n" + "=" * 50)
print("💡 KEY INSIGHTS:")
print("=" * 50)
survival_rate = df['Survived'].mean() * 100
print(f"Overall survival rate: {survival_rate:.2f}%")

class_survival = df.groupby('Pclass')['Survived'].mean() * 100
print(f"1st Class survival: {class_survival[1]:.2f}%")
print(f"2nd Class survival: {class_survival[2]:.2f}%")
print(f"3rd Class survival: {class_survival[3]:.2f}%")
print(f"Average age: {df['Age'].mean():.2f} years")
print(f"Average fare: {df['Fare'].mean():.2f}")

# ============ VISUALIZATION ============
print("\n📊 Generating visualization...")

classes = ['1st Class', '2nd Class', '3rd Class']
survival_rates = [class_survival[1], class_survival[2], class_survival[3]]

plt.figure(figsize=(8, 5))
bars = plt.bar(classes, survival_rates, color=['gold', 'silver', 'brown'])
plt.title('Week 3 — Titanic Survival Rate by Class')
plt.ylabel('Survival Rate %')
plt.ylim(0, 100)

for i, rate in enumerate(survival_rates):
    plt.text(i, rate + 1, f'{rate:.1f}%', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('results/week3_survival_chart.png')
plt.show()
print("✅ Chart saved to results/week3_survival_chart.png")
print("=" * 50) 