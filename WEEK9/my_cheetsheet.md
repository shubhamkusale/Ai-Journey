Text data           → Naive Bayes
Need explanation    → Decision Tree
Best accuracy       → Random Forest or XGBoost (coming)
Medical/high dim    → Logistic Regression or SVM
No labels           → K-Means or DBSCAN
Visualize data      → PCA first
Unknown K clusters  → DBSCAN

Customer_segment.py
df[['col1','col2']] → select multiple columns
StandardScaler → mean=0, std=1
fit = learn mean and std from data
transform = apply (value-mean)/std formula
fit_transform = both together
Inertia = sum of distances from each point to its centroid
Low inertia = tight clusters = good
High inertia = loose clusters = bad

## Matplotlib Basics
plt.figure(figsize=(w,h))  → create canvas
plt.plot(x, y, 'bo-')      → line chart
plt.bar(x, y)              → bar chart
plt.scatter(x, y)          → scatter plot
plt.xlabel('text')         → X axis label
plt.ylabel('text')         → Y axis label
plt.title('text')          → chart title
plt.legend()               → show legend
plt.savefig('name.png')    → save image (before show)
plt.show()                 → display chart
plt.tight_layout()         → fix overlapping elements

## Color codes
b=blue, r=red, g=green, k=black, y=yellow

## Marker codes  
o=circle, s=square, ^=triangle, *=star

## Line styles
- = solid, -- = dashed, : = dotted

StandardScaler mean=0 means: average value becomes zero
Values above average → positive
Values below average → negative