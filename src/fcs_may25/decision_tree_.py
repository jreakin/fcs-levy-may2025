from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from icecream import ic
import matplotlib.pyplot as plt

from fcs_may25.config import FilePaths
import protocols as p_


class FindlayDecisionTree:
    data: p_.ModelDataStartingPoint
    decision_features: list[str]
    X_decision: pd.DataFrame
    y_decision: pd.Series
    le: LabelEncoder
    tree: DecisionTreeClassifier
    feature_importance: pd.DataFrame
    def __init__(self, data: p_.ModelDataStartingPoint):
        self.data = data


    def setup_decision_tree(self):
        self.data.model_data['vote_decision'] = (self.data.model_data['nov_for_share'] >= 0.5).astype(int)
        self.decision_features = [
            'AGE_RANGE',
            'PARTY_CAT',
            'WARD',
            'PRECINCT_NAME',
            'AGE_WARD',
            'AGE_PRECINCT',
            'AGE_PARTY',
            'P_SCORE',
            'G_SCORE',
            'AGE'
        ]

    def preprocess_decision_tree(self):
        self.X_decision = self.data.model_data[self.decision_features].copy()
        self.y_decision = self.data.model_data['vote_decision']

        le = LabelEncoder()
        self.X_decision['AGE_RANGE'] = le.fit_transform(self.X_decision['AGE_RANGE'])
        self.X_decision['PARTY_CAT'] = le.fit_transform(self.X_decision['PARTY_CAT'])
        self.X_decision['WARD'] = le.fit_transform(self.X_decision['WARD'])
        self.X_decision['AGE_WARD'] = le.fit_transform(self.X_decision['AGE_WARD'])
        self.X_decision['AGE_PRECINCT'] = le.fit_transform(self.X_decision['AGE_PRECINCT'])
        self.X_decision['AGE_PARTY'] = le.fit_transform(self.X_decision['AGE_PARTY'])
        self.X_decision['PRECINCT_NAME'] = le.fit_transform(self.X_decision['PRECINCT_NAME'])

        self.tree = DecisionTreeClassifier(
            max_depth=5,  # Limit depth for interpretability
            min_samples_split=100,  # Minimum samples required to split
            min_samples_leaf=50,  # Minimum samples required at each leaf
            random_state=42
        )

        self.tree.fit(self.X_decision, self.y_decision)

        plt.figure(figsize=(20,10))
        plot_tree(
                self.tree,
                feature_names=self.decision_features,
                class_names=['Against', 'For'],
                filled=True,
                rounded=True,
                fontsize=10
            )
        plt.savefig(FilePaths.IMAGE_PATH / 'decision_tree.png')
        plt.show()
        plt.close()

        self.feature_importance = pd.DataFrame({
            'feature': self.decision_features,
            'importance': self.tree.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\nDecision Tree: Feature Importances:")
        print("=" * 50)
        print(self.feature_importance.to_markdown())
        print("-" * 50)

    @staticmethod
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
    
    def decision_path_analysis(self):
        path_info = self.analyze_decision_paths(self.tree, self.X_decision, self.decision_features)
        # Print some example paths
        print("\nDecision Tree: Example Decision Paths:")
        print("=" * 50)
        for i in range(min(5, len(path_info))):
            print(f"\tSample {i}:")
            print(f"\t\tPrediction: {'For' if path_info[i]['prediction'] == 1 else 'Against'}")
            print("\t\tDecision path:")
            for node in path_info[i]['path_nodes']:
                    if node != path_info[i]['leaf_node']:  # Skip leaf nodes
                        feature = self.decision_features[self.tree.tree_.feature[node]]
                        threshold = self.tree.tree_.threshold[node]
                        value = self.X_decision.iloc[i][feature]
                        direction = ">=" if value >= threshold else "<"
                        print(f"\t\t\t{feature} {direction} {threshold:.2f}")
            print("\t", "-" * 50)

    @staticmethod
    def create_sentiment_categories(prediction_probs, confidence_thresholds=None):
        """
        Create nuanced sentiment categories based on prediction probabilities.
        
        Args:
            prediction_probs: Array of prediction probabilities
            confidence_thresholds: Dictionary of thresholds for different sentiment levels
                                Default: {
                                    'strongly_against': 0.2,
                                    'moderately_against': 0.35,
                                    'slightly_against': 0.45,
                                    'neutral': 0.55,
                                    'slightly_for': 0.65,
                                    'moderately_for': 0.8,
                                    'strongly_for': 1.0
                                }
        """
        if confidence_thresholds is None:
            confidence_thresholds = {
                'strongly_against': 0.2,
                'moderately_against': 0.35,
                'slightly_against': 0.45,
                'neutral': 0.55,
                'slightly_for': 0.65,
                'moderately_for': 0.8,
                'strongly_for': 1.0
            }
        
        categories = []
        for prob in prediction_probs:
            if prob <= confidence_thresholds['strongly_against']:
                categories.append('strongly_against')
            elif prob <= confidence_thresholds['moderately_against']:
                categories.append('moderately_against')
            elif prob <= confidence_thresholds['slightly_against']:
                categories.append('slightly_against')
            elif prob <= confidence_thresholds['neutral']:
                categories.append('neutral')
            elif prob <= confidence_thresholds['slightly_for']:
                categories.append('slightly_for')
            elif prob <= confidence_thresholds['moderately_for']:
                categories.append('moderately_for')
            else:
                categories.append('strongly_for')
        
        return categories
    
    def run(self):
        self.setup_decision_tree()
        self.preprocess_decision_tree()
        self.decision_path_analysis()
        return self
    
    def run_sentiment_analysis(self):
        prediction_probs = self.data.model_data['P_for']
        sentiment_categories = self.create_sentiment_categories(prediction_probs)
        self.data.model_data['sentiment'] = sentiment_categories
        return self.data.model_data
    
    