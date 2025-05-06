from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from icecream import ic
import matplotlib.pyplot as plt
import numpy as np

from fcs_may25.config import FilePaths
import protocols as p_


class FindlayDecisionTree:
    data: p_.ModelDataStartingPoint
    decision_features: list[str]
    X_decision: pd.DataFrame
    y_decision: pd.Series
    le: LabelEncoder
    tree: RandomForestClassifier
    feature_importance: pd.DataFrame
    categorical_encoders: dict
    scaler: StandardScaler
    def __init__(self, data: p_.ModelDataStartingPoint):
        self.data = data
        self.categorical_encoders = {}
        self.scaler = StandardScaler()
        self.decision_features = [
            'AGE',
            'P_SCORE',
            'G_SCORE',
            'WARD',
            'PRECINCT_NAME',
            'PARTY_CAT',  # Changed from PARTY_AFFILIATION to PARTY_CAT
            'precinct_nov_for_share_mean',
            'ward_nov_for_share_mean',
            'precinct_vs_ward',
            'precinct_z_score',
            'ward_primary_ratio',  # Added primary turnout ratios
            'precinct_primary_ratio'
        ]


    def setup_decision_tree(self):
        # Calculate ward-level statistics
        ward_stats = self.data.model_data.groupby('WARD').agg({
            'nov_for_share': ['mean', 'std'],
            'P_SCORE': ['mean', 'std'],  # Primary election participation
            'G_SCORE': ['mean', 'std'],
            'P_SCORE_ALL': ['mean', 'std'],
            'G_SCORE_ALL': ['mean', 'std'],
            'AGE': ['mean', 'std']
        })
        ward_stats.columns = ['_'.join(col).strip() for col in ward_stats.columns.values]
        
        # Calculate precinct-level statistics
        precinct_stats = self.data.model_data.groupby('PRECINCT_NAME').agg({
            'nov_for_share': ['mean', 'std'],
            'P_SCORE': ['mean', 'std'],  # Primary election participation
            'G_SCORE': ['mean', 'std'],
            'P_SCORE_ALL': ['mean', 'std'],
            'G_SCORE_ALL': ['mean', 'std'],
            'AGE': ['mean', 'std']
        })
        precinct_stats.columns = ['_'.join(col).strip() for col in precinct_stats.columns.values]
        
        # Calculate primary turnout ratios
        ward_primary_ratios = self.data.model_data.groupby('WARD').apply(
            lambda x: x['P_SCORE'].mean() / x['nov_for_share'].mean()
        ).fillna(0.5)
        
        precinct_primary_ratios = self.data.model_data.groupby('PRECINCT_NAME').apply(
            lambda x: x['P_SCORE'].mean() / x['nov_for_share'].mean()
        ).fillna(0.5)
        
        # Add ward and precinct statistics to the data
        for col in ward_stats.columns:
            self.data.model_data[f'ward_{col}'] = self.data.model_data['WARD'].map(ward_stats[col])
            self.data.model_data[f'precinct_{col}'] = self.data.model_data['PRECINCT_NAME'].map(precinct_stats[col])
        
        # Add primary turnout ratios
        self.data.model_data['ward_primary_ratio'] = self.data.model_data['WARD'].map(ward_primary_ratios)
        self.data.model_data['precinct_primary_ratio'] = self.data.model_data['PRECINCT_NAME'].map(precinct_primary_ratios)
        
        # Calculate relative performance metrics
        self.data.model_data['precinct_vs_ward'] = (
            self.data.model_data['precinct_nov_for_share_mean'] - 
            self.data.model_data['ward_nov_for_share_mean']
        )
        
        # Calculate z-scores for precinct performance within ward
        self.data.model_data['precinct_z_score'] = (
            self.data.model_data['precinct_nov_for_share_mean'] - 
            self.data.model_data['ward_nov_for_share_mean']
        ) / self.data.model_data['ward_nov_for_share_std'].replace(0, 1)
        
        # Create target variable based on multiple factors including primary participation
        def categorize_performance(row):
            precinct_share = row['precinct_nov_for_share_mean']
            ward_share = row['ward_nov_for_share_mean']
            z_score = row['precinct_z_score']
            primary_participation = row['P_SCORE'] / 100.0  # Convert to probability
            precinct_primary_ratio = row['precinct_primary_ratio']
            ward_primary_ratio = row['ward_primary_ratio']
            
            # Adjust shares by primary turnout ratios
            adjusted_precinct_share = precinct_share * precinct_primary_ratio
            adjusted_ward_share = ward_share * ward_primary_ratio
            
            # More nuanced categorization based on both absolute and relative performance
            # and primary election participation
            if adjusted_precinct_share > 0.55 and primary_participation > 0.7:  # Strong performance and high primary participation
                if z_score > 1.5:
                    return 'strongly_for'
                elif z_score > 0.5:
                    return 'lean_for'
                else:
                    return 'swing_for'
            elif adjusted_precinct_share < 0.45 and primary_participation < 0.3:  # Weak performance and low primary participation
                if z_score < -1.5:
                    return 'strongly_against'
                elif z_score < -0.5:
                    return 'lean_against'
                else:
                    return 'swing_against'
            else:  # Middle range performance
                if z_score > 0.5 and primary_participation > 0.5:
                    return 'swing_for'
                elif z_score < -0.5 and primary_participation < 0.5:
                    return 'swing_against'
                else:
                    return 'swing'
        
        self.data.model_data['vote_decision'] = self.data.model_data.apply(categorize_performance, axis=1)
        
        # Define features that should be considered
        self.decision_features = [
            'AGE',
            'P_SCORE',  # Primary election participation
            'G_SCORE',
            'precinct_nov_for_share_mean',
            'ward_nov_for_share_mean',
            'precinct_vs_ward',
            'precinct_z_score',
            'PARTY_CAT',  # Changed from PARTY_AFFILIATION to PARTY_CAT
            'WARD',
            'PRECINCT_NAME',
            'ward_primary_ratio',  # Added primary turnout ratios
            'precinct_primary_ratio'
        ]

    def preprocess_decision_tree(self):
        self.X_decision = self.data.model_data[self.decision_features].copy()
        self.y_decision = self.data.model_data['vote_decision']

        # Fill any NaN values in numerical features with their means
        numerical_features = ['AGE', 'P_SCORE', 'G_SCORE', 'precinct_nov_for_share_mean', 
                            'ward_nov_for_share_mean', 'precinct_vs_ward', 'precinct_z_score']
        for feature in numerical_features:
            self.X_decision[feature] = self.X_decision[feature].fillna(self.X_decision[feature].mean())

        # Create party categories
        self.X_decision['PARTY_CAT'] = pd.Categorical(self.data.model_data['PARTY_CAT'], categories=['D', 'I', 'R'], ordered=True)

        # Create more sophisticated age groups based on voting patterns
        self.X_decision['AGE_DECADE'] = (self.X_decision['AGE'] // 10) * 10
        self.X_decision['AGE_GROUP'] = pd.qcut(self.X_decision['AGE'], q=5, labels=['very_young', 'young', 'middle', 'older', 'senior'])
        
        # Create party loyalty metrics
        self.X_decision['PARTY_LOYALTY'] = (self.X_decision['P_SCORE'] + self.X_decision['G_SCORE']) / 2
        self.X_decision['PARTY_STRENGTH'] = (self.X_decision['P_SCORE'] > 0.5).astype(int) + (self.X_decision['G_SCORE'] > 0.5).astype(int)
        self.X_decision['VOTING_CONSISTENCY'] = abs(self.X_decision['P_SCORE'] - self.X_decision['G_SCORE'])
        
        # Create voting pattern categories
        self.X_decision['VOTING_PATTERN'] = (
            (self.X_decision['P_SCORE'] > 0.5).astype(str) + '_' + 
            (self.X_decision['G_SCORE'] > 0.5).astype(str)
        )
        
        # Create relative performance metrics
        self.X_decision['PERFORMANCE_GAP'] = abs(self.X_decision['precinct_nov_for_share_mean'] - self.X_decision['ward_nov_for_share_mean'])
        self.X_decision['RELATIVE_PERFORMANCE'] = self.X_decision['precinct_nov_for_share_mean'] / self.X_decision['ward_nov_for_share_mean']
        
        # Create interaction features
        self.X_decision['AGE_PARTY_INTERACTION'] = self.X_decision['AGE_DECADE'] * self.X_decision['PARTY_LOYALTY']
        self.X_decision['VOTING_HISTORY'] = self.X_decision['P_SCORE'] + self.X_decision['G_SCORE']
        self.X_decision['VOTING_FREQUENCY'] = (self.X_decision['P_SCORE'] > 0).astype(int) + (self.X_decision['G_SCORE'] > 0).astype(int)
        
        # Create precinct-level interaction features
        self.X_decision['PRECINCT_AGE_MEAN'] = self.X_decision.groupby('PRECINCT_NAME')['AGE'].transform('mean')
        self.X_decision['PRECINCT_PARTY_MEAN'] = self.X_decision.groupby('PRECINCT_NAME')['PARTY_LOYALTY'].transform('mean')
        self.X_decision['AGE_VS_PRECINCT'] = self.X_decision['AGE'] - self.X_decision['PRECINCT_AGE_MEAN']
        self.X_decision['PARTY_VS_PRECINCT'] = self.X_decision['PARTY_LOYALTY'] - self.X_decision['PRECINCT_PARTY_MEAN']
        
        # Handle categorical features
        categorical_features = ['PARTY_CAT', 'WARD', 'PRECINCT_NAME', 'AGE_GROUP', 'VOTING_PATTERN']
        for col in categorical_features:
            self.X_decision[col] = pd.Categorical(self.X_decision[col])
            if 'UNKNOWN' not in self.X_decision[col].cat.categories:
                self.X_decision[col] = self.X_decision[col].cat.add_categories('UNKNOWN')
            self.X_decision[col] = self.X_decision[col].fillna('UNKNOWN')
            
            if col not in self.categorical_encoders:
                self.categorical_encoders[col] = LabelEncoder()
                self.categorical_encoders[col].fit(self.X_decision[col].astype(str))
            self.X_decision[col] = self.categorical_encoders[col].transform(self.X_decision[col].astype(str))

        # Store the feature order
        self.feature_order = self.X_decision.columns.tolist()

        # Normalize numerical features
        numerical_cols = self.X_decision.select_dtypes(include=['float64', 'int64']).columns
        self.X_decision[numerical_cols] = self.scaler.fit_transform(self.X_decision[numerical_cols])

        # Create and fit the Random Forest with adjusted parameters
        self.tree = RandomForestClassifier(
            n_estimators=100,  # Number of trees
            max_depth=15,  # Increased depth to allow more complex splits
            min_samples_split=30,  # Reduced to allow more splits
            min_samples_leaf=15,  # Reduced to allow more granular leaves
            min_impurity_decrease=0.0005,  # Allow smaller improvements
            random_state=42,
            class_weight='balanced',
            max_features='sqrt',  # Consider sqrt of total features at each split
            min_weight_fraction_leaf=0.01,  # Allow smaller leaves
            bootstrap=True,  # Use bootstrap samples
            oob_score=True  # Calculate out-of-bag score
        )

        self.tree.fit(self.X_decision, self.y_decision)

        # Print category distribution
        print("\nRandom Forest: Vote Decision Category Distribution:")
        print("=" * 50)
        dist = pd.Series(self.y_decision).value_counts()
        print(dist.to_markdown())
        print("\nRandom Forest: Category Descriptions:")
        print("strongly_for: High absolute performance (>60%) AND significantly better than ward")
        print("lean_for: Good performance (>50%) OR better than ward average")
        print("swing: Close to ward average and moderate absolute performance")
        print("lean_against: Poor performance (<50%) OR worse than ward average")
        print("strongly_against: Low absolute performance (<40%) AND significantly worse than ward")
        
        # Calculate and display feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_order,
            'importance': self.tree.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\nRandom Forest: Feature Importances:")
        print("=" * 50)
        print(self.feature_importance.to_markdown())
        print("-" * 50)
        
        # Create visualization of feature importance
        plt.figure(figsize=(12, 8))
        plt.barh(self.feature_importance['feature'], self.feature_importance['importance'])
        plt.xlabel('Feature Importance')
        plt.title('Random Forest Feature Importance')
        plt.tight_layout()
        plt.savefig(FilePaths.IMAGE_PATH / 'random_forest_feature_importance.png')
        plt.close()
        
        # Print out-of-bag score
        print(f"\nOut-of-bag score: {self.tree.oob_score_:.4f}")

    @staticmethod
    def analyze_decision_paths(forest, X, feature_names):
        """Analyze decision paths across all trees in the forest"""
        # Get predictions from all trees
        all_predictions = []
        for tree in forest.estimators_:
            predictions = tree.predict(X)
            all_predictions.append(predictions)
        
        # Calculate prediction distribution
        prediction_dist = {}
        for i in range(len(X)):
            precinct_predictions = [pred[i] for pred in all_predictions]
            pred_counts = {}
            for pred in precinct_predictions:
                pred_counts[pred] = pred_counts.get(pred, 0) + 1
            prediction_dist[i] = pred_counts
        
        return prediction_dist
    
    def decision_path_analysis(self):
        """Analyze decision paths and feature importance across the forest"""
        # Get prediction distribution
        prediction_dist = self.analyze_decision_paths(self.tree, self.X_decision, self.decision_features)
        
        # Group predictions by precinct
        precinct_paths = {}
        for i, pred_counts in prediction_dist.items():
            precinct = self.data.model_data.iloc[i]['PRECINCT_NAME']
            if precinct not in precinct_paths:
                precinct_paths[precinct] = []
            precinct_paths[precinct].append(pred_counts)
        
        # Print precinct-specific analysis
        print("\nRandom Forest: Precinct-Level Analysis:")
        print("=" * 50)
        for precinct, pred_counts_list in precinct_paths.items():
            print(f"\nPrecinct: {precinct}")
            print(f"Number of samples: {len(pred_counts_list)}")
            
            # Aggregate predictions across all samples in precinct
            total_preds = {}
            for pred_counts in pred_counts_list:
                for pred, count in pred_counts.items():
                    total_preds[pred] = total_preds.get(pred, 0) + count
            
            # Print prediction distribution
            print("\nPrediction Distribution:")
            total_samples = sum(total_preds.values())
            for pred, count in sorted(total_preds.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_samples) * 100
                print(f"\t{pred}: {count} ({percentage:.1f}%)")
            print("-" * 50)
        
        # Print overall feature importance
        print("\nRandom Forest: Overall Feature Importances:")
        print("=" * 50)
        feature_importance = pd.DataFrame({
            'feature': self.feature_order,
            'importance': self.tree.feature_importances_
        }).sort_values('importance', ascending=False)
        print(feature_importance.to_markdown())
        
        # Print feature importance by tree
        print("\nFeature Importance by Tree:")
        print("=" * 50)
        tree_importances = np.array([tree.feature_importances_ for tree in self.tree.estimators_])
        mean_importance = tree_importances.mean(axis=0)
        std_importance = tree_importances.std(axis=0)
        
        importance_df = pd.DataFrame({
            'feature': self.feature_order,
            'mean_importance': mean_importance,
            'std_importance': std_importance
        }).sort_values('mean_importance', ascending=False)
        print(importance_df.head(10).to_markdown())
        
        return self

    @staticmethod
    def create_sentiment_categories(prediction_probs):
        """
        Create balanced sentiment categories using percentile-based thresholds.
        
        Args:
            prediction_probs: Array of actual vote shares (0-1 scale)
        """
        # Calculate percentile thresholds for even distribution
        thresholds = pd.Series(prediction_probs).quantile([0.2, 0.4, 0.6, 0.8]).tolist()
        
        categories = []
        for prob in prediction_probs:
            if prob <= thresholds[0]:
                categories.append('strongly_against')
            elif prob <= thresholds[1]:
                categories.append('lean_against')
            elif prob <= thresholds[2]:
                categories.append('swing')
            elif prob <= thresholds[3]:
                categories.append('lean_for')
            else:
                categories.append('strongly_for')
        
        return categories
    
    def validate_against_results(self):
        """Validate decision tree predictions against actual November results."""
        # Calculate ward-level November results
        ward_results = self.data.model_data.groupby('WARD')['nov_for_share'].mean()
        
        # Calculate precinct-level predictions with dynamic weights based on November results
        precinct_predictions = []
        for precinct, group in self.data.model_data.groupby('PRECINCT_NAME'):
            # Get the ward's November performance
            ward = group['WARD'].iloc[0]
            ward_nov_share = ward_results[ward]
            
            # Calculate decision distribution for this precinct
            decisions = group['vote_decision'].value_counts(normalize=True)
            
            # Adjust weights based on ward's November performance
            if ward_nov_share < 0.3:
                # Strongly against ward
                weights = {
                    'strongly_for': 0.1,
                    'lean_for': 0.2,
                    'swing_for': 0.3,
                    'swing': 0.4,
                    'swing_against': 0.5,
                    'lean_against': 0.7,
                    'strongly_against': 0.9
                }
            elif ward_nov_share < 0.4:
                # Swing against ward
                weights = {
                    'strongly_for': 0.2,
                    'lean_for': 0.3,
                    'swing_for': 0.4,
                    'swing': 0.5,
                    'swing_against': 0.6,
                    'lean_against': 0.7,
                    'strongly_against': 0.8
                }
            elif ward_nov_share < 0.6:
                # Swing ward
                weights = {
                    'strongly_for': 0.3,
                    'lean_for': 0.4,
                    'swing_for': 0.5,
                    'swing': 0.5,
                    'swing_against': 0.5,
                    'lean_against': 0.6,
                    'strongly_against': 0.7
                }
            elif ward_nov_share < 0.7:
                # Swing for ward
                weights = {
                    'strongly_for': 0.4,
                    'lean_for': 0.5,
                    'swing_for': 0.6,
                    'swing': 0.5,
                    'swing_against': 0.4,
                    'lean_against': 0.3,
                    'strongly_against': 0.2
                }
            else:
                # Strongly for ward
                weights = {
                    'strongly_for': 0.7,
                    'lean_for': 0.6,
                    'swing_for': 0.5,
                    'swing': 0.4,
                    'swing_against': 0.3,
                    'lean_against': 0.2,
                    'strongly_against': 0.1
                }
            
            # Calculate weighted prediction
            predicted_for_share = sum(
                decisions.get(category, 0) * weight 
                for category, weight in weights.items()
            )
            
            precinct_predictions.append({
                'precinct': precinct,
                'ward': ward,
                'predicted_for_share': predicted_for_share,
                'nov_for_share': group['nov_for_share'].mean()
            })
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(precinct_predictions)
        comparison_df['change'] = comparison_df['predicted_for_share'] - comparison_df['nov_for_share']
        
        # Calculate ward-level results
        ward_comparison = comparison_df.groupby('ward').agg({
            'predicted_for_share': 'mean',
            'nov_for_share': 'mean',
            'change': 'mean'
        }).reset_index()
        
        # Format the results
        ward_comparison['predicted_for_share'] = ward_comparison['predicted_for_share'].map('{:.1%}'.format)
        ward_comparison['nov_for_share'] = ward_comparison['nov_for_share'].map('{:.1%}'.format)
        ward_comparison['change'] = ward_comparison['change'].map('{:.1%}'.format)
        
        # Sort by predicted_for_share
        ward_comparison = ward_comparison.sort_values('predicted_for_share', ascending=False)
        
        # Print ward-level results
        print("\nWard-Level Results:")
        print("-" * 50)
        print(ward_comparison.to_markdown(index=False))
        print("-" * 50)
        
        return comparison_df

    def run(self):
        self.setup_decision_tree()
        self.preprocess_decision_tree()
        self.decision_path_analysis()
        validation_results = self.validate_against_results()
        return self
    
    def run_sentiment_analysis(self):
        # Use actual vote share for categorization
        prediction_probs = self.data.model_data['nov_for_share']
        
        # Create balanced categories
        sentiment_categories = self.create_sentiment_categories(prediction_probs)
        self.data.model_data['dt_sentiment'] = sentiment_categories
        
        # Print distribution and thresholds
        print("\nSentiment Category Distribution:")
        print("=" * 50)
        dist = pd.Series(sentiment_categories).value_counts()
        print(dist.to_markdown())
        
        # Print threshold information
        thresholds = pd.Series(prediction_probs).quantile([0.2, 0.4, 0.6, 0.8])
        print("\nCategory Thresholds:")
        print("=" * 50)
        print(f"strongly_against: <= {thresholds[0.2]:.2%}")
        print(f"lean_against: {thresholds[0.2]:.2%} - {thresholds[0.4]:.2%}")
        print(f"swing: {thresholds[0.4]:.2%} - {thresholds[0.6]:.2%}")
        print(f"lean_for: {thresholds[0.6]:.2%} - {thresholds[0.8]:.2%}")
        print(f"strongly_for: > {thresholds[0.8]:.2%}")
        
        print(f"\nTotal samples: {len(sentiment_categories)}")
        print(f"Category ratios:")
        for cat in dist.index:
            print(f"{cat}: {dist[cat]/len(sentiment_categories):.2%}")
        
        return self.data.model_data
    
    