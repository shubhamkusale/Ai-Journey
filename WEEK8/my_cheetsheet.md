so in this week we will build heart disease analyzer 

1 - what is feature scaling 

feature scaling means aur age is 18 and aur salay is 20000 so these numbers are so much big that bigger number will dominate so we will do normalization of these numbers converting Converting to the SAME RANGE 

age 18  → 0.22  (18 out of max 80)
fare 500 → 1.0  (maximum fare)
fare 7   → 0.014 (minimum fare)
pclass 1 → 0.0
pclass 3 → 1.0
like these 

## sklearn warehouse map
preprocessing  → MinMaxScaler, StandardScaler
model_selection → train_test_split
linear_model   → LogisticRegression
ensemble       → RandomForestClassifier
tree           → DecisionTreeClassifier
neighbors      → KNeighborsClassifier
svm            → SVC
metrics        → accuracy_score, classification_report
naive_bayes    → MultinomialNB
feature_extraction.text → CountVectorizer

Your Instinct Was Right
Next time with a new dataset — first step is always:
1. What is this data about?
2. What does each column mean?
3. What are we trying to predict?
4. What's the class distribution?