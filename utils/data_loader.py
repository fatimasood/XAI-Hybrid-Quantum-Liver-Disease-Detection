import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

from utils.config import DATA_PATH, PLOTS_DIR, RANDOM_SEED, FEATURE_NAMES

class DataLoader:
    def __init__(self, filepath="data/Indian Liver Patient Dataset (ILPD).csv"):
        self.filepath = filepath
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        
    def load_and_preprocess(self):
        self.df = pd.read_csv(self.filepath)
        
        df_cleaned = self.df.copy()
        
        # Categorical variable
        df_cleaned['Gender'] = df_cleaned['Gender'].map({'Male': 0, 'Female': 1})
        
        # Binary output
        df_cleaned['Sickness'] = df_cleaned['Sickness'].replace(2, 0)
        
        # NAN values
        df_cleaned['A/G'] = df_cleaned['A/G'].fillna(df_cleaned['A/G'].mean())
        
        # Store cleaned dataframe
        self.df = df_cleaned
        self.feature_names = [col for col in df_cleaned.columns if col != 'Sickness']
        
        # Filter positive and negative cases
        pos = df_cleaned[df_cleaned['Sickness'] == 1]
        neg = df_cleaned[df_cleaned['Sickness'] == 0]
        
        # Determine number of test samples
        n_test = min(len(pos), len(neg)) // 2
        
        # Sample test data
        test_pos = pos.sample(n=n_test, random_state=42)
        test_neg = neg.sample(n=n_test, random_state=42)
        
        # Combine into balanced test set
        test = pd.concat([test_pos, test_neg]).reset_index(drop=True)
        
        # Indices to remove
        drop_idx = [3, 31, 35, 89, 104, 106, 114, 115, 116, 124, 130, 132, 135, 139, 143, 150, 151, 157, 161]
        print(test.loc[drop_idx])
        test = test.drop(drop_idx, errors='ignore').reset_index(drop=True)
        
        # Remaining data for training
        train = df_cleaned.drop(test.index).reset_index(drop=True)
        
        # Remove duplicates
        test = test.drop_duplicates().reset_index(drop=True)
        train = train.drop_duplicates().reset_index(drop=True)
        
        # Separate features and labels
        self.X_train, self.y_train = train.drop(columns=['Sickness']), train['Sickness']
        self.X_test, self.y_test = test.drop(columns=['Sickness']), test['Sickness']
        
        return self.scale_features()
    
    def scale_features(self):
        """Scale features using StandardScaler"""
        
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

        self.y_train = self.y_train.to_numpy()
        self.y_test = self.y_test.to_numpy()
        
        return self
    
    def get_class_weights(self):
        """Calculate class weights for imbalanced data"""
        from sklearn.utils import class_weight
        
        class_weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(self.y_train),
            y=self.y_train
        )
        
        return dict(enumerate(class_weights))
    
    def perform_eda(self):
        """Perform exploratory data analysis and save plots"""
        
        # Class distribution
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        
        class_counts = self.df['Sickness'].value_counts()
        axes[0, 0].bar(['No Disease', 'Disease'], class_counts.values, 
                       color=['#2ecc71', '#e74c3c'])
        axes[0, 0].set_title('Class Distribution', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Count')
        
        # Gender distribution
        gender_class = pd.crosstab(self.df['Gender'], self.df['Sickness'])
        gender_class.plot(kind='bar', ax=axes[0, 1], color=['#2ecc71', '#e74c3c'])
        axes[0, 1].set_title('Gender Distribution by Class', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Gender (0=Male, 1=Female)')
        axes[0, 1].legend(['No Disease', 'Disease'])
        
        # Age distribution
        axes[0, 2].hist([self.df[self.df['Sickness']==0]['Age'], 
                         self.df[self.df['Sickness']==1]['Age']], 
                        bins=20, label=['No Disease', 'Disease'], 
                        color=['#2ecc71', '#e74c3c'], alpha=0.7)
        axes[0, 2].set_title('Age Distribution by Class', fontsize=14, fontweight='bold')
        axes[0, 2].set_xlabel('Age')
        axes[0, 2].legend()
        
        # Correlation matrix
        corr_matrix = self.df.corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                    ax=axes[1, 0], cbar_kws={'label': 'Correlation'})
        axes[1, 0].set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
        
        # Feature distributions
        features = ['TB', 'DB', 'Alkphos', 'Sgpt']
        for i, feature in enumerate(features[:2]):
            self.df.boxplot(column=feature, by='Sickness', ax=axes[1, i+1])
            axes[1, i+1].set_title(f'{feature} Distribution')
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'eda_overview.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Interactive plotly dashboard
        self._create_interactive_dashboard()
        
        return self
    
    def _create_interactive_dashboard(self):
        """Create interactive Plotly dashboard"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Feature Distributions', 'Class Balance', 
                          'Age Distribution', 'Correlation Heatmap')
        )
        
        # Box plots for features
        features = ['TB', 'DB', 'Alkphos', 'Sgpt']
        for feature in features:
            for disease_status in [0, 1]:
                values = self.df[self.df['Sickness'] == disease_status][feature]
                fig.add_trace(
                    go.Box(y=values, name=f'{feature} (Class {disease_status})', 
                          showlegend=False),
                    row=1, col=1
                )
        
        # Class balance
        class_counts = self.df['Sickness'].value_counts()
        fig.add_trace(
            go.Bar(x=['No Disease', 'Disease'], y=class_counts.values,
                   marker_color=['#2ecc71', '#e74c3c']),
            row=1, col=2
        )
        
        # Age histogram
        for disease_status, color in zip([0, 1], ['#2ecc71', '#e74c3c']):
            fig.add_trace(
                go.Histogram(x=self.df[self.df['Sickness'] == disease_status]['Age'], 
                           name=f'Class {disease_status}', marker_color=color),
                row=2, col=1
            )
        
        # Correlation heatmap
        corr_matrix = self.df.corr()
        fig.add_trace(
            go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, 
                       y=corr_matrix.columns, colorscale='RdBu_r', zmid=0),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text="Interactive EDA Dashboard")
        fig.write_html(os.path.join(PLOTS_DIR, 'interactive_eda.html'))
        
        return fig
    
    def get_data_summary(self):
        """Print data summary"""
        
        print("\n..... DATA SUMMARY .....")
        print(f"Total samples: {len(self.df)}")
        print(f"Features: {len(self.feature_names)}")
        print(f"Classes: 0 (No Disease), 1 (Disease)")
        print(f"Class distribution:\n{self.df['Sickness'].value_counts()}")
        print(f"\nTraining set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
        print(f"Missing values: {self.df.isnull().sum().sum()}")
