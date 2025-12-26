#!/usr/bin/env python3
"""Quick demo script for patient readmission prediction."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data.utils import generate_synthetic_readmission_data, DataProcessor, create_patient_level_split
from models.tabular import ModelFactory, get_model_config
from metrics.clinical import ModelEvaluator
from utils.core import set_seed

def main():
    """Run a quick demo of the patient readmission prediction system."""
    print("🏥 Patient Readmission Prediction Demo")
    print("=" * 50)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Generate synthetic data
    print("\n1. Generating synthetic data...")
    df = generate_synthetic_readmission_data(n_samples=500, n_patients=250)
    print(f"   Generated {len(df)} samples with {df['readmitted'].mean():.1%} readmission rate")
    
    # Create patient-level splits
    print("\n2. Creating patient-level splits...")
    train_df, val_df, test_df = create_patient_level_split(
        df, patient_id_col='patient_id', test_size=0.2, val_size=0.1
    )
    print(f"   Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Process data
    print("\n3. Processing data...")
    processor = DataProcessor(
        categorical_features=['admission_type', 'discharge_disposition', 'diag_primary', 'gender'],
        numerical_features=['age', 'length_of_stay', 'num_medications', 'num_diagnoses'],
        target_column='readmitted'
    )
    
    X_train, y_train = processor.fit_transform(train_df)
    X_test = processor.transform(test_df)
    y_test = test_df['readmitted'].values
    
    print(f"   Features: {X_train.shape[1]}, Samples: {X_train.shape[0]}")
    
    # Train models
    print("\n4. Training models...")
    models_to_test = ['xgboost', 'lightgbm', 'random_forest']
    results = []
    
    for model_type in models_to_test:
        print(f"   Training {model_type}...")
        model_config = get_model_config(model_type)
        model = ModelFactory.create_model(model_type, **model_config)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        evaluator = ModelEvaluator()
        result = evaluator.evaluate(y_test, y_pred_proba, model_name=model_type)
        results.append(result)
        
        print(f"     AUC-ROC: {result['metrics']['auc_roc']:.3f}")
    
    # Show best model
    best_result = max(results, key=lambda x: x['metrics']['auc_roc'])
    print(f"\n5. Best model: {best_result['model_name']}")
    print(f"   AUC-ROC: {best_result['metrics']['auc_roc']:.3f}")
    print(f"   Sensitivity: {best_result['metrics']['sensitivity']:.3f}")
    print(f"   Specificity: {best_result['metrics']['specificity']:.3f}")
    
    print("\n✅ Demo completed successfully!")
    print("\n⚠️  Remember: This is a research demonstration tool only.")
    print("   NOT FOR CLINICAL USE. NOT MEDICAL ADVICE.")

if __name__ == "__main__":
    main()
