import pickle
import joblib
import json
import numpy as np
import pandas as pd
from datetime import datetime
import os
import warnings

class ResultsSaver:
    """
    Comprehensive class to save and load ML experiment results
    """
    
    def __init__(self, base_path="ml_results"):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
    
    def save_complete_results(self, results, experiment_name=None):
        """
        Save complete results including models, metrics, and data
        
        Parameters:
        -----------
        results : dict
            Complete results dictionary from train_and_evaluate_models_with_calibration
        experiment_name : str
            Name for this experiment (if None, uses timestamp)
        
        Returns:
        --------
        str : Path where results were saved
        """
        if experiment_name is None:
            experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        experiment_path = os.path.join(self.base_path, experiment_name)
        os.makedirs(experiment_path, exist_ok=True)
        
        print(f"Saving results to: {experiment_path}")
        
        # Save different components separately
        saved_components = {}
        
        for model_name, model_results in results.items():
            model_path = os.path.join(experiment_path, model_name)
            os.makedirs(model_path, exist_ok=True)
            
            # 1. Save models using joblib (recommended for sklearn models)
            try:
                joblib.dump(model_results['model'], 
                           os.path.join(model_path, 'model.joblib'))
                joblib.dump(model_results['calibrated_model'], 
                           os.path.join(model_path, 'calibrated_model.joblib'))
                saved_components[f'{model_name}_models'] = 'joblib'
            except Exception as e:
                print(f"Warning: Could not save models for {model_name} with joblib: {e}")
                try:
                    # Fallback to pickle
                    with open(os.path.join(model_path, 'model.pkl'), 'wb') as f:
                        pickle.dump(model_results['model'], f)
                    with open(os.path.join(model_path, 'calibrated_model.pkl'), 'wb') as f:
                        pickle.dump(model_results['calibrated_model'], f)
                    saved_components[f'{model_name}_models'] = 'pickle'
                except Exception as e2:
                    print(f"Error: Could not save models for {model_name}: {e2}")
                    saved_components[f'{model_name}_models'] = 'failed'
            
            # 2. Save numpy arrays
            np.save(os.path.join(model_path, 'predictions.npy'), 
                    model_results['predictions'])
            np.save(os.path.join(model_path, 'probabilities_raw.npy'), 
                    model_results['probabilities_raw'])
            np.save(os.path.join(model_path, 'probabilities_calibrated.npy'), 
                    model_results['probabilities_calibrated'])
            
            # 3. Save metrics and other serializable data
            serializable_data = {
                'cv_f1_mean': float(model_results['cv_f1_mean']),
                'cv_f1_std': float(model_results['cv_f1_std']),
                'test_accuracy': float(model_results['test_accuracy']),
                'test_precision': float(model_results['test_precision']),
                'test_recall': float(model_results['test_recall']),
                'test_f1': float(model_results['test_f1']),
                'test_auc_raw': float(model_results.get('test_auc_raw', np.nan)),
                'test_auc_calibrated': float(model_results.get('test_auc_calibrated', np.nan)),
            }
            
            with open(os.path.join(model_path, 'metrics.json'), 'w') as f:
                json.dump(serializable_data, f, indent=2)
            
            # 4. Save calibration metrics
            self._save_calibration_metrics(model_results['calibration_metrics'], model_path)
            
            # 5. Save risk scores
            self._save_risk_scores(model_results['risk_scores'], model_path)
            
            # 6. Save feature importance
            self._save_feature_importance(model_results['feature_importance'], model_path)
        
        # Save experiment metadata
        metadata = {
            'experiment_name': experiment_name,
            'timestamp': datetime.now().isoformat(),
            'saved_components': saved_components,
            'models_included': list(results.keys())
        }
        
        with open(os.path.join(experiment_path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Results saved successfully to: {experiment_path}")
        return experiment_path
    
    def _save_calibration_metrics(self, cal_metrics, model_path):
        """Save calibration metrics"""
        cal_path = os.path.join(model_path, 'calibration')
        os.makedirs(cal_path, exist_ok=True)
        
        # Save scalar metrics
        scalar_metrics = {
            'brier_score_raw': float(cal_metrics.get('brier_score_raw', np.nan)),
            'brier_score_calibrated': float(cal_metrics.get('brier_score_calibrated', np.nan)),
            'brier_improvement': float(cal_metrics.get('brier_improvement', np.nan))
        }
        
        with open(os.path.join(cal_path, 'scalar_metrics.json'), 'w') as f:
            json.dump(scalar_metrics, f, indent=2)
        
        # Save calibration curves
        if cal_metrics.get('calibration_curve_raw') is not None:
            np.save(os.path.join(cal_path, 'cal_curve_raw_fraction.npy'),
                    cal_metrics['calibration_curve_raw']['fraction_positives'])
            np.save(os.path.join(cal_path, 'cal_curve_raw_mean.npy'),
                    cal_metrics['calibration_curve_raw']['mean_predicted'])
        
        if cal_metrics.get('calibration_curve_calibrated') is not None:
            np.save(os.path.join(cal_path, 'cal_curve_cal_fraction.npy'),
                    cal_metrics['calibration_curve_calibrated']['fraction_positives'])
            np.save(os.path.join(cal_path, 'cal_curve_cal_mean.npy'),
                    cal_metrics['calibration_curve_calibrated']['mean_predicted'])
    
    def _save_risk_scores(self, risk_scores, model_path):
        """Save risk scores"""
        risk_path = os.path.join(model_path, 'risk_scores')
        os.makedirs(risk_path, exist_ok=True)
        
        # Save arrays
        np.save(os.path.join(risk_path, 'max_probability.npy'), 
                risk_scores['max_probability'])
        np.save(os.path.join(risk_path, 'predicted_class.npy'), 
                risk_scores['predicted_class'])
        np.save(os.path.join(risk_path, 'class_probabilities.npy'), 
                risk_scores['class_probabilities'])
        
        # Save categorical data
        pd.Series(risk_scores['risk_category']).to_csv(
            os.path.join(risk_path, 'risk_category.csv'), index=False
        )
        
        # Save risk performance if available
        if 'risk_performance' in risk_scores:
            with open(os.path.join(risk_path, 'risk_performance.json'), 'w') as f:
                # Convert numpy types to Python types
                perf_data = {}
                for category, metrics in risk_scores['risk_performance'].items():
                    perf_data[category] = {
                        'count': int(metrics['count']),
                        'accuracy': float(metrics['accuracy']),
                        'mean_probability': float(metrics['mean_probability'])
                    }
                json.dump(perf_data, f, indent=2)
    
    def _save_feature_importance(self, feat_importance, model_path):
        """Save feature importance data"""
        if not feat_importance:
            return
        
        feat_path = os.path.join(model_path, 'feature_importance')
        os.makedirs(feat_path, exist_ok=True)
        
        # Save as DataFrame for easy loading
        if 'importance_df' in feat_importance:
            feat_importance['importance_df'].to_csv(
                os.path.join(feat_path, 'importance.csv'), index=False
            )
        
        # Save SHAP data if available
        if 'shap_values' in feat_importance and feat_importance['shap_values'] is not None:
            np.save(os.path.join(feat_path, 'shap_values.npy'), 
                    feat_importance['shap_values'])
        
        # Save other importance data (excluding DataFrames and non-serializable objects)
        other_data = {}
        for k, v in feat_importance.items():
            if k not in ['importance_df', 'shap_values']:
                try:
                    if isinstance(v, np.ndarray):
                        other_data[k] = v.tolist()
                    elif isinstance(v, pd.DataFrame):
                        # Save DataFrames separately
                        v.to_csv(os.path.join(feat_path, f'{k}.csv'), index=False)
                        other_data[f'{k}_info'] = f'DataFrame saved as {k}.csv'
                    elif isinstance(v, pd.Series):
                        # Save Series as CSV
                        v.to_csv(os.path.join(feat_path, f'{k}.csv'), index=True, header=[k])
                        other_data[f'{k}_info'] = f'Series saved as {k}.csv'
                    elif isinstance(v, (int, float, str, bool, list, dict)):
                        other_data[k] = v
                    elif hasattr(v, 'tolist') and callable(getattr(v, 'tolist')):
                        # For numpy scalars and similar
                        other_data[k] = v.tolist()
                    else:
                        # Skip non-serializable objects
                        other_data[f'{k}_type'] = str(type(v))
                        other_data[f'{k}_info'] = 'Non-serializable object skipped'
                except Exception as e:
                    print(f"Warning: Could not save feature importance item '{k}': {e}")
                    other_data[f'{k}_error'] = str(e)
        
        if other_data:
            with open(os.path.join(feat_path, 'other_importance.json'), 'w') as f:
                json.dump(other_data, f, indent=2)
    
    def load_complete_results(self, experiment_path):
        """
        Load complete results from saved experiment
        
        Parameters:
        -----------
        experiment_path : str
            Path to saved experiment
        
        Returns:
        --------
        dict : Reconstructed results dictionary
        """
        print(f"Loading results from: {experiment_path}")
        
        # Load metadata
        with open(os.path.join(experiment_path, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
        
        print(f"Experiment: {metadata['experiment_name']}")
        print(f"Timestamp: {metadata['timestamp']}")
        print(f"Models: {metadata['models_included']}")
        
        results = {}
        
        for model_name in metadata['models_included']:
            model_path = os.path.join(experiment_path, model_name)
            results[model_name] = self._load_model_results(model_path, metadata['saved_components'])
        
        print("✅ Results loaded successfully!")
        return results
    
    def _load_model_results(self, model_path, saved_components):
        """Load individual model results"""
        model_results = {}
        
        # Load models
        model_key = f"{os.path.basename(model_path)}_models"
        if saved_components.get(model_key) == 'joblib':
            try:
                model_results['model'] = joblib.load(os.path.join(model_path, 'model.joblib'))
                model_results['calibrated_model'] = joblib.load(os.path.join(model_path, 'calibrated_model.joblib'))
            except:
                print(f"Warning: Could not load joblib models from {model_path}")
        elif saved_components.get(model_key) == 'pickle':
            try:
                with open(os.path.join(model_path, 'model.pkl'), 'rb') as f:
                    model_results['model'] = pickle.load(f)
                with open(os.path.join(model_path, 'calibrated_model.pkl'), 'rb') as f:
                    model_results['calibrated_model'] = pickle.load(f)
            except:
                print(f"Warning: Could not load pickle models from {model_path}")
        
        # Load numpy arrays
        model_results['predictions'] = np.load(os.path.join(model_path, 'predictions.npy'))
        model_results['probabilities_raw'] = np.load(os.path.join(model_path, 'probabilities_raw.npy'))
        model_results['probabilities_calibrated'] = np.load(os.path.join(model_path, 'probabilities_calibrated.npy'))
        
        # Load metrics
        with open(os.path.join(model_path, 'metrics.json'), 'r') as f:
            metrics = json.load(f)
            model_results.update(metrics)
        
        # Load calibration metrics
        model_results['calibration_metrics'] = self._load_calibration_metrics(model_path)
        
        # Load risk scores
        model_results['risk_scores'] = self._load_risk_scores(model_path)
        
        # Load feature importance
        model_results['feature_importance'] = self._load_feature_importance(model_path)
        
        return model_results
    
    def _load_calibration_metrics(self, model_path):
        """Load calibration metrics"""
        cal_path = os.path.join(model_path, 'calibration')
        cal_metrics = {}
        
        # Load scalar metrics
        try:
            with open(os.path.join(cal_path, 'scalar_metrics.json'), 'r') as f:
                cal_metrics.update(json.load(f))
        except:
            pass
        
        # Load calibration curves
        try:
            cal_metrics['calibration_curve_raw'] = {
                'fraction_positives': np.load(os.path.join(cal_path, 'cal_curve_raw_fraction.npy')),
                'mean_predicted': np.load(os.path.join(cal_path, 'cal_curve_raw_mean.npy'))
            }
        except:
            cal_metrics['calibration_curve_raw'] = None
        
        try:
            cal_metrics['calibration_curve_calibrated'] = {
                'fraction_positives': np.load(os.path.join(cal_path, 'cal_curve_cal_fraction.npy')),
                'mean_predicted': np.load(os.path.join(cal_path, 'cal_curve_cal_mean.npy'))
            }
        except:
            cal_metrics['calibration_curve_calibrated'] = None
        
        return cal_metrics
    
    def _load_risk_scores(self, model_path):
        """Load risk scores"""
        risk_path = os.path.join(model_path, 'risk_scores')
        risk_scores = {}
        
        try:
            risk_scores['max_probability'] = np.load(os.path.join(risk_path, 'max_probability.npy'))
            risk_scores['predicted_class'] = np.load(os.path.join(risk_path, 'predicted_class.npy'))
            risk_scores['class_probabilities'] = np.load(os.path.join(risk_path, 'class_probabilities.npy'))
            
            risk_scores['risk_category'] = pd.read_csv(
                os.path.join(risk_path, 'risk_category.csv')
            ).iloc[:, 0].values
            
            # Load risk performance
            try:
                with open(os.path.join(risk_path, 'risk_performance.json'), 'r') as f:
                    risk_scores['risk_performance'] = json.load(f)
            except:
                pass
        except Exception as e:
            print(f"Warning: Could not load risk scores: {e}")
        
        return risk_scores
    
    def _load_feature_importance(self, model_path):
        """Load feature importance"""
        feat_path = os.path.join(model_path, 'feature_importance')
        feat_importance = {}
        
        try:
            # Load importance DataFrame
            feat_importance['importance_df'] = pd.read_csv(
                os.path.join(feat_path, 'importance.csv')
            )
            
            # Load SHAP values
            try:
                feat_importance['shap_values'] = np.load(
                    os.path.join(feat_path, 'shap_values.npy')
                )
            except:
                pass
            
            # Load other importance data
            try:
                with open(os.path.join(feat_path, 'other_importance.json'), 'r') as f:
                    other_data = json.load(f)
                    
                    # Load any DataFrames that were saved separately
                    for key in list(other_data.keys()):
                        if key.endswith('_info') and 'DataFrame saved as' in str(other_data[key]):
                            csv_filename = key.replace('_info', '') + '.csv'
                            try:
                                feat_importance[key.replace('_info', '')] = pd.read_csv(
                                    os.path.join(feat_path, csv_filename)
                                )
                            except:
                                pass
                        elif key.endswith('_info') and 'Series saved as' in str(other_data[key]):
                            csv_filename = key.replace('_info', '') + '.csv'
                            try:
                                feat_importance[key.replace('_info', '')] = pd.read_csv(
                                    os.path.join(feat_path, csv_filename), 
                                    index_col=0
                                ).iloc[:, 0]
                            except:
                                pass
                        elif not key.endswith(('_info', '_type', '_error')):
                            feat_importance[key] = other_data[key]
                            
            except Exception as e:
                print(f"Warning: Could not load other importance data: {e}")
                
        except Exception as e:
            print(f"Warning: Could not load feature importance: {e}")
        
        return feat_importance
    
    def create_results_summary(self, results, save_path=None):
        """
        Create a human-readable summary of results
        """
        summary = []
        summary.append("="*80)
        summary.append("ML EXPERIMENT RESULTS SUMMARY")
        summary.append("="*80)
        summary.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary.append(f"Number of models: {len(results)}")
        summary.append("")
        
        # Create comparison table
        comparison_data = []
        for name, result in results.items():
            comparison_data.append({
                'Model': name,
                'CV F1 Score': f"{result['cv_f1_mean']:.4f} ± {result['cv_f1_std']:.4f}",
                'Test Accuracy': f"{result['test_accuracy']:.4f}",
                'Test F1 Score': f"{result['test_f1']:.4f}",
                'AUC (Calibrated)': f"{result.get('test_auc_calibrated', 'N/A'):.4f}" if not np.isnan(result.get('test_auc_calibrated', np.nan)) else 'N/A',
                'Brier Score Improvement': f"{result['calibration_metrics'].get('brier_improvement', 'N/A'):.4f}" if not np.isnan(result['calibration_metrics'].get('brier_improvement', np.nan)) else 'N/A'
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        summary.append("MODEL COMPARISON:")
        summary.append(df_comparison.to_string(index=False))
        summary.append("")
        
        # Best performing model
        best_f1_model = max(results.keys(), key=lambda k: results[k]['test_f1'])
        summary.append(f"🏆 BEST PERFORMING MODEL (F1 Score): {best_f1_model}")
        summary.append("")
        
        # Calibration summary
        summary.append("CALIBRATION IMPROVEMENTS:")
        for name, result in results.items():
            cal_metrics = result['calibration_metrics']
            improvement = cal_metrics.get('brier_improvement', np.nan)
            if not np.isnan(improvement):
                summary.append(f"  {name}: {improvement:.4f} (Brier score reduction)")
        
        summary_text = "\n".join(summary)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(summary_text)
            print(f"Summary saved to: {save_path}")
        
        return summary_text


# Usage example functions
def save_results_example(results):
    """
    Example of how to use the ResultsSaver
    """
    # Create saver instance
    saver = ResultsSaver(base_path="my_ml_experiments")
    
    # Save complete results
    experiment_path = saver.save_complete_results(
        results, 
        experiment_name="calibrated_classification_experiment"
    )
    
    # Create and save summary
    summary = saver.create_results_summary(
        results, 
        save_path=os.path.join(experiment_path, "results_summary.txt")
    )
    
    print("\n" + summary)
    
    return experiment_path

def load_results_example(experiment_path):
    """
    Example of how to load saved results
    """
    saver = ResultsSaver()
    
    # Load complete results
    loaded_results = saver.load_complete_results(experiment_path)
    
    # You can now use loaded_results just like the original results
    return loaded_results

# Quick save function for convenience
def quick_save_results(results, experiment_name=None):
    """
    One-liner to save results with default settings
    """
    saver = ResultsSaver()
    return saver.save_complete_results(results, experiment_name)