
So week6 project is about learning decision Treeand The Random Forest using the Titanic Data
  
In this project we used 3 different ml model for getting accurate answers on aur Titanic Data using 1 :- KNN, 2 :- Decision Tree, 3 :- Random Forest 
These are the Scores :-  
Decision Tree: 0.7988826815642458
Random Forest: 0.8268156424581006
KNN          : 0.7150837988826816

So as we saw the three Models in which the Random Forest won because it has more Trees in it so it can get more feateures more unique answers 

In all this my brain got surprised by the models we have like Random Forest which trains data in different 100 trees to get a single answer thats really great because 1 tree can make problem but 100 cant

## Top Features That Predicted Survival

From the Random Forest feature importance chart:
1. fare - 27% - richer passengers had better access to lifeboats
2. sex - 26% - women were prioritized during evacuation  
3. age - 25% - children were also prioritized

## What I Learned
- 60% of ML work is cleaning data before model even runs
- One Decision Tree overfits but 100 trees voting together cancels mistakes
- Feature importance tells you WHY the model made decisions not just what
- Random Forest beat Decision Tree by 3% and KNN by 11%