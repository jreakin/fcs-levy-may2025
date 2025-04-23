from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from data_loader import ElectionReconstruction
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings

# Load the data
election_reconstruction = ElectionReconstruction()
data = election_reconstruction.data
records = data.data
election_results = data.election_results

# Merge the data
new_data = records.merge(election_results, right_on="precinct", left_on="PRECINCT_NAME", how="left")

# Prepare features for the decision tree
# We'll use simpler features that are more interpretable
features = [
    'AGE_RANGE',
    'PARTY_AFFILIATION',
    'WARD',
    'P_SCORE',
    'G_SCORE',
    'AGE'
]

# Create the target variable (binary for simplicity)
new_data['vote_decision'] = (new_data['nov_for_share'] >= 0.5).astype(int)

# Prepare X and y
X = new_data[features].copy()
y = new_data['vote_decision']

# Encode categorical variables
le = LabelEncoder()
X['AGE_RANGE'] = le.fit_transform(X['AGE_RANGE'])
X['PARTY_AFFILIATION'] = le.fit_transform(X['PARTY_AFFILIATION'])
X['WARD'] = le.fit_transform(X['WARD'])

# Create and fit the decision tree
tree = DecisionTreeClassifier(
    max_depth=5,  # Limit depth for interpretability
    min_samples_split=100,  # Minimum samples required to split
    min_samples_leaf=50,  # Minimum samples required at each leaf
    random_state=42
)

tree.fit(X, y)

# Visualize the decision tree
plt.figure(figsize=(20,10))
plot_tree(
    tree,
    feature_names=features,
    class_names=['Against', 'For'],
    filled=True,
    rounded=True,
    fontsize=10
)
plt.savefig('decision_tree.png', dpi=300, bbox_inches='tight')
plt.close()

# Print feature importances
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': tree.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importances:")
print(feature_importance)

# Function to analyze decision paths
def analyze_decision_paths(tree, X, feature_names):
    # Get the decision paths
    paths = tree.decision_path(X)
    
    # Get leaf node assignments
    leaves = tree.apply(X)
    
    # Create a DataFrame to store path information
    path_info = []
    
    for i in range(len(X)):
        # Get the path for this sample
        path = paths.getrow(i)
        path_nodes = path.indices
        
        # Get the leaf node
        leaf = leaves[i]
        
        # Get the prediction
        pred = tree.predict([X.iloc[i]])[0]
        
        # Store the information
        path_info.append({
            'sample_index': i,
            'path_nodes': path_nodes,
            'leaf_node': leaf,
            'prediction': pred
        })
    
    return path_info

# Analyze decision paths
path_info = analyze_decision_paths(tree, X, features)

# Print some example paths
print("\nExample Decision Paths:")
for i in range(min(5, len(path_info))):
    print(f"\nSample {i}:")
    print(f"Prediction: {'For' if path_info[i]['prediction'] == 1 else 'Against'}")
    print("Decision path:")
    for node in path_info[i]['path_nodes']:
        if node != path_info[i]['leaf_node']:  # Skip leaf nodes
            feature = features[tree.tree_.feature[node]]
            threshold = tree.tree_.threshold[node]
            value = X.iloc[i][feature]
            direction = ">=" if value >= threshold else "<"
            print(f"  {feature} {direction} {threshold:.2f}")

def print_readable_path(tree, X, sample_idx, feature_names):
    """Print a decision path in a readable format."""
    # Get the path for this sample
    path = tree.decision_path(X).getrow(sample_idx)
    path_nodes = path.indices
    
    # Get the leaf node
    leaf = tree.apply(X)[sample_idx]
    
    # Get the prediction
    pred = tree.predict([X.iloc[sample_idx]])[0]
    
    print(f"\nDecision Path for Sample {sample_idx}:")
    print(f"Final Prediction: {'For' if pred == 1 else 'Against'}")
    print("\nDecision Rules:")
    
    # Print each decision in the path
    for node in path_nodes:
        if node != leaf:  # Skip leaf nodes
            feature = feature_names[tree.tree_.feature[node]]
            threshold = tree.tree_.threshold[node]
            value = X.iloc[sample_idx][feature]
            direction = ">=" if value >= threshold else "<"
            print(f"  IF {feature} {direction} {threshold:.2f}")
            print(f"    (Current value: {value:.2f})")
            print(f"    THEN go {'right' if value >= threshold else 'left'}")

# Print a few example paths in readable format
print("\nExample Decision Paths (Readable Format):")
for i in range(3):  # Print first 3 samples
    print_readable_path(tree, X, i, features)
    print("-" * 50)

# Test different max_iter values
def test_max_iter_impact(X, y, max_iter_values=[100, 500, 1000, 2000]):
    """Test the impact of different max_iter values on model accuracy."""
    results = []
    
    # Split the data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    for max_iter in max_iter_values:
        # Create and fit the model
        model = LogisticRegression(max_iter=max_iter, random_state=42)
        
        # Catch convergence warnings
        with warnings.catch_warnings(record=True) as w:
            model.fit(X_train, y_train)
            has_warning = any(issubclass(warn.category, UserWarning) for warn in w)
        
        # Make predictions
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Store results
        results.append({
            'max_iter': max_iter,
            'accuracy': accuracy,
            'converged': not has_warning,
            'n_iter': model.n_iter_[0] if hasattr(model, 'n_iter_') else None
        })
    
    # Create DataFrame with results
    results_df = pd.DataFrame(results)
    print("\nImpact of max_iter on Model Performance:")
    print(results_df.to_string(index=False))
    
    return results_df

# Run the test
max_iter_results = test_max_iter_impact(X, y) 