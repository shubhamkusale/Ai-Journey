import numpy as np
import matplotlib.pyplot as plt 

# ============ LOAD DATA ============
data = np.genfromtxt(r'c:\Users\Shubham\Desktop\Ai-Journey\week2\numpy-data-analyzer\tested.csv', 
                      delimiter=',', skip_header=1)

# ============ EXTRACT COLUMNS ============
Survived = data[:, 1]
Pclass =   data[:, 2]
Age =      data[:, 6]
Fare =     data[:, 10]

print("=" * 40)
print("🤖 JARVIS TITANIC ANALYSIS")
print("=" * 40)

print(f"total Passangers: {len(data)}")
print(f"Survival Rate: {np.nanmean(Survived)*100:.2f}%")
print(f"Average Age: {np.nanmean(Age):.2f}")
print(f"Average Fate: {np.nanmean(Fare):.2f}")
print(f"Min Fate: {np.nanmin(Fare):.2f}")
print(f"Max Fate: {np.nanmax(Fare):.2f}")

Class1 = Survived[Pclass == 1]  
Class2 = Survived[Pclass == 2]  
Class3 = Survived[Pclass == 3]  

print(f"1st Class Survival Rate: {np.nanmean(Class1)*100:.2f}%")
print(f"2nd Class Survival Rate: {np.nanmean(Class2)*100:.2f}%")
print(f"3rd Class Survival Rate: {np.nanmean(Class3)*100:.2f}%")

print("=" * 40)


classes = ['1st Class', '2nd Class', '3rd Class']
survival_rates = [
    np.nanmean(Class1)*100,
    np.nanmean(Class2)*100,
    np.nanmean(Class3)*100
]

plt.figure(figsize=(8, 5))
plt.bar(classes, survival_rates, color=['gold', 'silver', 'brown'])
plt.title('🤖 Jarvis: Titanic Survival Rate by Class')
plt.ylabel('Survival Rate %')
plt.ylim(0, 100)

for i, rate in enumerate(survival_rates):
    plt.text(i, rate + 1, f'{rate:.1f}%', ha='center')

plt.savefig('results/survival_chart.png')
plt.show()
print("Chart saved to results folder!")