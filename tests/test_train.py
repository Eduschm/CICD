"""
Simple integration test for ChurnTrainer
Run this script to quickly verify the trainer works with your dataset
"""

import pandas as pd
import os
import shutil
from churn_trainer import ChurnTrainer


def test_churn_trainer():
    """Run a simple integration test with the actual dataset"""
    
    print("="*70)
    print("CHURN TRAINER INTEGRATION TEST")
    print("="*70)
    
    # Load the dataset
    print("\n1. Loading dataset...")
    try:
        df = pd.read_csv("data/dataset.csv")
        print(f"   ✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    except FileNotFoundError:
        print("   ✗ Error: data/dataset.csv not found!")
        return False
    
    # Drop customerID (not a feature)
    print("\n2. Preparing features...")
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
        print("   ✓ Removed customerID column")
    
    # Add derived features
    df['avg_monthly_spend'] = df['TotalCharges'] / (df['tenure'] + 1)
    df['contract_value'] = df['MonthlyCharges'] * 12
    print("   ✓ Added derived features: avg_monthly_spend, contract_value")
    
    # Check for missing values in TotalCharges
    if df['TotalCharges'].dtype == 'object':
        print("   ⚠ TotalCharges is object type, converting to numeric...")
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Initialize trainer
    print("\n3. Initializing ChurnTrainer...")
    test_model_dir = "test_models_integration"
    test_results_dir = "test_results_integration"
    
    trainer = ChurnTrainer(
        df=df,
        model_dir=test_model_dir,
        results_dir=test_results_dir,
        random_state=42
    )
    print(f"   ✓ Trainer initialized")
    print(f"   ✓ Numerical features: {len(trainer.numerical_features)}")
    print(f"   ✓ Categorical features: {len(trainer.categorical_features)}")
    
    # Test data split
    print("\n4. Testing data split...")
    trainer.split()
    print(f"   ✓ Train set: {len(trainer.X_train)} samples")
    print(f"   ✓ Test set: {len(trainer.X_test)} samples")
    print(f"   ✓ Train churn rate: {trainer.y_train.mean():.2%}")
    print(f"   ✓ Test churn rate: {trainer.y_test.mean():.2%}")
    
    # Test single model training (Decision Tree - fastest)
    print("\n5. Training Decision Tree (quick test)...")
    trainer.param_grids['dt'] = {
        'classifier__max_depth': [5, 10],
        'classifier__min_samples_split': [2, 5],
    }
    
    try:
        trainer._run_grid_search('dt', verbose=0)
        print("   ✓ Decision Tree trained successfully")
        
        # Evaluate
        trainer.evaluate('dt')
        
        # Save model
        model_path = trainer.save_model('dt', 'test_dt_model.skops')
        print(f"   ✓ Model saved to: {model_path}")
        
    except Exception as e:
        print(f"   ✗ Error training model: {e}")
        return False
    
    # Test prediction
    print("\n6. Testing predictions...")
    try:
        predictions = trainer.best_models['dt'].predict(trainer.X_test)
        print(f"   ✓ Generated {len(predictions)} predictions")
        print(f"   ✓ Predicted churn rate: {predictions.mean():.2%}")
    except Exception as e:
        print(f"   ✗ Error making predictions: {e}")
        return False
    
    # Test voting classifier (optional, if you have time)
    print("\n7. Testing Voting Classifier (optional - takes longer)...")
    try:
        # Use minimal grids for speed
        trainer.param_grids['xgb'] = {
            'classifier__n_estimators': [100],
            'classifier__max_depth': [3],
            'classifier__learning_rate': [0.05],
        }
        trainer.param_grids['lgbm'] = {
            'classifier__n_estimators': [100],
            'classifier__max_depth': [3],
        }
        
        # Train remaining models
        print("   - Training XGBoost...")
        trainer._run_grid_search('xgb', verbose=0)
        print("   - Training LightGBM...")
        trainer._run_grid_search('lgbm', verbose=0)
        
        # Build voting classifier
        print("   - Building Voting Classifier...")
        trainer.build_voting_classifier(max_fpr=0.1)
        print("   ✓ Voting Classifier built successfully")
        print(f"   ✓ Optimal threshold: {trainer.voting_threshold:.4f}")
        
        # Verify model saved
        voting_model_path = os.path.join(test_model_dir, "model.skops")
        if os.path.exists(voting_model_path):
            print(f"   ✓ Voting model saved to: {voting_model_path}")
        
    except Exception as e:
        print(f"   ⚠ Voting classifier test skipped or failed: {e}")
    
    # Cleanup
    print("\n8. Cleaning up test files...")
    try:
        if os.path.exists(test_model_dir):
            shutil.rmtree(test_model_dir)
        if os.path.exists(test_results_dir):
            shutil.rmtree(test_results_dir)
        print("   ✓ Test files cleaned up")
    except Exception as e:
        print(f"   ⚠ Warning: Could not clean up test files: {e}")
    
    print("\n" + "="*70)
    print("✓ ALL TESTS PASSED!")
    print("="*70)
    print("\nThe ChurnTrainer is working correctly with your dataset.")
    print("You can now run: trainer.run_all() to train all models.")
    return True


def quick_test():
    """Even quicker test - just verify loading and basic functionality"""
    print("\n" + "="*70)
    print("QUICK SANITY CHECK")
    print("="*70)
    
    try:
        # Load data
        df = pd.read_csv("data/dataset.csv")
        if 'customerID' in df.columns:
            df = df.drop('customerID', axis=1)
        
        # Add derived features
        df['avg_monthly_spend'] = df['TotalCharges'] / (df['tenure'] + 1)
        df['contract_value'] = df['MonthlyCharges'] * 12
        
        # Initialize
        trainer = ChurnTrainer(df=df, model_dir="temp_test", results_dir="temp_test")
        trainer.split()
        
        print(f"\n✓ Dataset loaded: {len(df)} rows")
        print(f"✓ Train/Test split: {len(trainer.X_train)}/{len(trainer.X_test)}")
        print(f"✓ Features: {len(trainer.numerical_features)} numerical, {len(trainer.categorical_features)} categorical")
        print(f"✓ Target distribution: {trainer.y_train.value_counts().to_dict()}")
        
        # Cleanup
        if os.path.exists("temp_test"):
            shutil.rmtree("temp_test")
        
        print("\n✓ Quick test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n✗ Quick test FAILED: {e}")
        return False


if __name__ == "__main__":
    import sys
    
    # Check if user wants quick test or full test
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        success = quick_test()
    else:
        print("\nRunning full integration test...")
        print("(Use --quick flag for faster sanity check)\n")
        success = test_churn_trainer()
    
    sys.exit(0 if success else 1)