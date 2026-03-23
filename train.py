import os
import numpy as np
import pandas as pd
import cv2
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, VGG16, EfficientNetB0, InceptionV3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class BrainTumorHybridClassifier:
    def __init__(self, base_path="Br35H-Mask-RCNN"):
        self.base_path = base_path
        self.yes_path = os.path.join(base_path, "yes")
        self.no_path = os.path.join(base_path, "no")
        self.img_size = (224, 224)
        self.batch_size = 32
        
    def load_complete_dataset(self):
        
        all_images = []
        all_labels = []
        
    
        
        
        if os.path.exists(self.no_path):
            no_files = [f for f in os.listdir(self.no_path) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            print(f"Found {len(no_files)} no-tumor images in 'no' folder")
            
            for img_file in no_files:
                img_path = os.path.join(self.no_path, img_file)
                try:
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.resize(img, self.img_size)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        all_images.append(img)
                        all_labels.append(0) 
                except Exception as e:
                    print(f"Error loading {img_file}: {e}")
                    continue
        else:
            print(f"Warning: 'no' folder not found at {self.no_path}")
        
        
        if os.path.exists(self.yes_path):
            yes_files = [f for f in os.listdir(self.yes_path) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            print(f"Found {len(yes_files)} tumor images in 'yes' folder")
            
            for img_file in yes_files:
                img_path = os.path.join(self.yes_path, img_file)
                try:
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.resize(img, self.img_size)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        all_images.append(img)
                        all_labels.append(1) 
                except Exception as e:
                    print(f"Error loading {img_file}: {e}")
                    continue
        else:
            print(f"Warning: 'yes' folder not found at {self.yes_path}")
        
        print(f"Total images loaded: {len(all_images)}")
        print(f"Tumor images: {sum(all_labels)}")
        print(f"No-tumor images: {len(all_labels) - sum(all_labels)}")
        
        return np.array(all_images), np.array(all_labels)
    
    def create_cnn_feature_extractor(self, architecture='resnet50'):
    
        try:
            if architecture == 'resnet50':
                base_model = ResNet50(weights='imagenet', include_top=False, 
                                    input_shape=(224, 224, 3))
            elif architecture == 'vgg16':
                base_model = VGG16(weights='imagenet', include_top=False,
                                 input_shape=(224, 224, 3))
            elif architecture == 'efficientnet':
                base_model = EfficientNetB0(weights='imagenet', include_top=False,
                                          input_shape=(224, 224, 3))
            elif architecture == 'inception':
                base_model = InceptionV3(weights='imagenet', include_top=False,
                                       input_shape=(224, 224, 3))
            else:
                raise ValueError(f"Unsupported architecture: {architecture}")
            
        
            base_model.trainable = False
            
    
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dense(512, activation='relu')(x)
            x = Dropout(0.5)(x)
            feature_output = Dense(256, activation='relu', name='features')(x)
            
    
            feature_extractor = Model(inputs=base_model.input, outputs=feature_output)
            
            print(f" {architecture.upper()} feature extractor created successfully")
            return feature_extractor
            
        except Exception as e:
            print(f" Error creating {architecture} feature extractor: {e}")
            return None
    
    def extract_features(self, images, feature_extractor):
        """Extract features from images using CNN"""
        if len(images) == 0:
            print("No images to extract features from!")
            return np.array([])
        
        if feature_extractor is None:
            print("Feature extractor is None!")
            return np.array([])
        
        
        images = images.astype('float32') / 255.0
        
        
        features = []
        total_batches = (len(images) + self.batch_size - 1) // self.batch_size
        
        for i in range(0, len(images), self.batch_size):
            batch = images[i:i+self.batch_size]
            batch_num = i // self.batch_size + 1
            
            try:
                batch_features = feature_extractor.predict(batch, verbose=0)
                features.append(batch_features)
                
                if batch_num % 5 == 0 or batch_num == total_batches:
                    print(f"Processed batch {batch_num}/{total_batches}")
                    
            except Exception as e:
                print(f"Error extracting features for batch {batch_num}: {e}")
                return np.array([])
        
        if features:
            return np.vstack(features)
        else:
            return np.array([])
    
    def train_ml_classifiers(self, X_train, y_train, X_val, y_val):
    
        if len(X_train) == 0 or len(X_val) == 0:
            print("No training or validation data available!")
            return {}, {}
        
    
        print(f"Training set - Tumor: {sum(y_train)}, No-tumor: {len(y_train) - sum(y_train)}")
        print(f"Validation set - Tumor: {sum(y_val)}, No-tumor: {len(y_val) - sum(y_val)}")
        
    
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        
        classifiers = {
            'Random Forest': RandomForestClassifier(
                n_estimators=200, 
                max_depth=10, 
                min_samples_split=5,
                random_state=42
            ),
            'SVM (RBF)': SVC(
                kernel='rbf', 
                C=1.0, 
                gamma='scale',
                probability=True, 
                random_state=42
            ),
            'SVM (Linear)': SVC(
                kernel='linear', 
                C=1.0,
                probability=True, 
                random_state=42
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=200, 
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            ),
            'Logistic Regression': LogisticRegression(
                C=1.0,
                random_state=42, 
                max_iter=1000
            )
        }
        
        results = {}
        trained_models = {}
        
        for name, clf in classifiers.items():
            print(f"\n Training {name}...")
            
            try:
            
                clf.fit(X_train_scaled, y_train)
                
            
                y_val_pred = clf.predict(X_val_scaled)
                y_val_proba = clf.predict_proba(X_val_scaled)[:, 1] if hasattr(clf, 'predict_proba') else None
                
    
                accuracy = accuracy_score(y_val, y_val_pred)
                
                results[name] = {
                    'accuracy': accuracy,
                    'predictions': y_val_pred,
                    'probabilities': y_val_proba,
                    'classification_report': classification_report(y_val, y_val_pred, output_dict=True)
                }
                
                trained_models[name] = {'model': clf, 'scaler': scaler}
                
                print(f" {name} Validation Accuracy: {accuracy:.4f}")
                
            except Exception as e:
                print(f" Error training {name}: {e}")
                continue
        
        return results, trained_models
    
    def evaluate_pure_dl(self, X_train, y_train, X_val, y_val, X_test, y_test, architecture='resnet50'):
        """Create and evaluate pure DL model for comparison"""
        try:
            if architecture == 'resnet50':
                base_model = ResNet50(weights='imagenet', include_top=False, 
                                    input_shape=(224, 224, 3))
            elif architecture == 'vgg16':
                base_model = VGG16(weights='imagenet', include_top=False,
                                 input_shape=(224, 224, 3))
            elif architecture == 'efficientnet':
                base_model = EfficientNetB0(weights='imagenet', include_top=False,
                                          input_shape=(224, 224, 3))
            
            # Freeze base model initially
            base_model.trainable = False
            
            # Add classification head
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dense(512, activation='relu')(x)
            x = Dropout(0.5)(x)
            x = Dense(256, activation='relu')(x)
            x = Dropout(0.3)(x)
            predictions = Dense(1, activation='sigmoid')(x)
            
            model = Model(inputs=base_model.input, outputs=predictions)
            
            # Compile model
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            # Prepare callbacks
            early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7)
            
            # Normalize data
            X_train_norm = X_train.astype('float32') / 255.0
            X_val_norm = X_val.astype('float32') / 255.0
            X_test_norm = X_test.astype('float32') / 255.0
            
            print(f" Training Pure {architecture.upper()} DL model...")
            
            # Train model
            history = model.fit(
                X_train_norm, y_train,
                validation_data=(X_val_norm, y_val),
                epochs=20,
                batch_size=self.batch_size,
                callbacks=[early_stopping, reduce_lr],
                verbose=1
            )
            
            # Evaluate on test set
            test_predictions = model.predict(X_test_norm)
            test_predictions_binary = (test_predictions > 0.5).astype(int).flatten()
            test_accuracy = accuracy_score(y_test, test_predictions_binary)
            
            print(f" Pure {architecture.upper()} DL Test Accuracy: {test_accuracy:.4f}")
            
            return {
                'accuracy': test_accuracy,
                'predictions': test_predictions_binary,
                'probabilities': test_predictions.flatten(),
                'model': model,
                'history': history
            }
            
        except Exception as e:
            print(f" Error training Pure {architecture} DL: {e}")
            return None
    
    def run_comprehensive_evaluation(self):
        """Run comprehensive evaluation with proper balanced dataset"""
        print(" Brain Tumor Detection - Comprehensive Hybrid ML+DL Analysis")
        print("="*70)
        
        # Load complete balanced dataset
        X_combined, y_combined = self.load_complete_dataset()
        
        if len(X_combined) == 0:
            print(" No data loaded! Please check your dataset paths.")
            return {}
        
        # Check if we have both classes
        unique_classes = np.unique(y_combined)
        if len(unique_classes) < 2:
            print(" Dataset contains only one class! Need both tumor and no-tumor images.")
            print(f"Available classes: {unique_classes}")
            return {}
        
        print(f" Balanced dataset loaded with {len(unique_classes)} classes")
        
        # Create stratified train/val/test splits
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_combined, y_combined, test_size=0.2, random_state=42, stratify=y_combined
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
        )
        
        print(f"\n Dataset Splits:")
        print(f"Train: {len(X_train)} ({sum(y_train)} tumor, {len(y_train)-sum(y_train)} no-tumor)")
        print(f"Val: {len(X_val)} ({sum(y_val)} tumor, {len(y_val)-sum(y_val)} no-tumor)")
        print(f"Test: {len(X_test)} ({sum(y_test)} tumor, {len(y_test)-sum(y_test)} no-tumor)")
        
        # Test different CNN architectures
        architectures = ['resnet50', 'vgg16', 'efficientnet']
        all_results = {}
        
        for arch in architectures:
            print(f"\n{'='*70}")
            print(f"🔬 Testing {arch.upper()} Architecture")
            print(f"{'='*70}")
            
            try:
                # Create feature extractor
                feature_extractor = self.create_cnn_feature_extractor(arch)
                if feature_extractor is None:
                    continue
                
                # Extract features
                print(" Extracting features from training set...")
                train_features = self.extract_features(X_train, feature_extractor)
                
                print(" Extracting features from validation set...")
                val_features = self.extract_features(X_val, feature_extractor)
                
                print(" Extracting features from test set...")
                test_features = self.extract_features(X_test, feature_extractor)
                
                if len(train_features) == 0:
                    print(f" Feature extraction failed for {arch}")
                    continue
                
                print(f" Features extracted: {train_features.shape}")
                
                # Train ML classifiers on extracted features
                ml_results, trained_models = self.train_ml_classifiers(
                    train_features, y_train, val_features, y_val
                )
                
                # Evaluate on test set
                test_results = {}
                for ml_name, ml_data in trained_models.items():
                    test_features_scaled = ml_data['scaler'].transform(test_features)
                    test_pred = ml_data['model'].predict(test_features_scaled)
                    test_acc = accuracy_score(y_test, test_pred)
                    
                    # Get classification report
                    test_report = classification_report(y_test, test_pred, output_dict=True)
                    
                    test_results[ml_name] = {
                        'test_accuracy': test_acc,
                        'predictions': test_pred,
                        'classification_report': test_report
                    }
                    
                    print(f" {ml_name} Test Accuracy: {test_acc:.4f}")
                
                # Store results for this architecture
                all_results[arch] = {
                    'validation_results': ml_results,
                    'test_results': test_results
                }
                
                # Evaluate pure DL for comparison
                print(f"\n Evaluating Pure {arch.upper()} DL approach...")
                pure_dl_result = self.evaluate_pure_dl(
                    X_train, y_train, X_val, y_val, X_test, y_test, arch
                )
                
                if pure_dl_result:
                    all_results[arch]['pure_dl'] = pure_dl_result
                
            except Exception as e:
                print(f" Error with {arch}: {e}")
                continue
        
        return all_results
    
    def display_comprehensive_results(self, all_results):
        """Display comprehensive results with detailed analysis"""
        if not all_results:
            print(" No results to display!")
            return None
        
        print("\n" + "="*80)
        print("🏆 COMPREHENSIVE BRAIN TUMOR DETECTION RESULTS")
        print("="*80)
        
        results_data = []
        
        # Collect all results
        for arch, arch_results in all_results.items():
            # Hybrid approaches
            if 'test_results' in arch_results:
                for ml_name, ml_result in arch_results['test_results'].items():
                    results_data.append({
                        'Architecture': arch.upper(),
                        'Approach': f'{arch.upper()} + {ml_name}',
                        'Type': 'Hybrid DL+ML',
                        'Test_Accuracy': ml_result['test_accuracy'],
                        'Precision': ml_result['classification_report']['1']['precision'],
                        'Recall': ml_result['classification_report']['1']['recall'],
                        'F1_Score': ml_result['classification_report']['1']['f1-score']
                    })
            
            # Pure DL approach
            if 'pure_dl' in arch_results:
                results_data.append({
                    'Architecture': arch.upper(),
                    'Approach': f'Pure {arch.upper()}',
                    'Type': 'Pure DL',
                    'Test_Accuracy': arch_results['pure_dl']['accuracy'],
                    'Precision': 'N/A',
                    'Recall': 'N/A',
                    'F1_Score': 'N/A'
                })
        
        if not results_data:
            print(" No valid results found!")
            return None
        
        # Sort by accuracy
        results_data.sort(key=lambda x: x['Test_Accuracy'], reverse=True)
        
        # Display ranked results
        print(f"\n RANKED RESULTS (by Test Accuracy):")
        print("-" * 80)
        print(f"{'Rank':<5} {'Approach':<30} {'Type':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10}")
        print("-" * 80)
        
        for idx, result in enumerate(results_data, 1):
            precision = f"{result['Precision']:.4f}" if result['Precision'] != 'N/A' else 'N/A'
            recall = f"{result['Recall']:.4f}" if result['Recall'] != 'N/A' else 'N/A'
            
            print(f"{idx:<5} {result['Approach']:<30} {result['Type']:<15} "
                  f"{result['Test_Accuracy']:<10.4f} {precision:<10} {recall:<10}")
        
        # Best performing approaches
        best = results_data[0]
        best_hybrid = next((r for r in results_data if r['Type'] == 'Hybrid DL+ML'), None)
        best_pure_dl = next((r for r in results_data if r['Type'] == 'Pure DL'), None)
        
        print(f"\n BEST OVERALL: {best['Approach']} ({best['Test_Accuracy']:.4f} accuracy)")
        
        if best_hybrid:
            print(f" BEST HYBRID: {best_hybrid['Approach']} ({best_hybrid['Test_Accuracy']:.4f} accuracy)")
        
        if best_pure_dl:
            print(f" BEST PURE DL: {best_pure_dl['Approach']} ({best_pure_dl['Test_Accuracy']:.4f} accuracy)")
        
        # Analysis insights
        print(f"\n PERFORMANCE INSIGHTS:")
        hybrid_results = [r for r in results_data if r['Type'] == 'Hybrid DL+ML']
        pure_dl_results = [r for r in results_data if r['Type'] == 'Pure DL']
        
        if hybrid_results and pure_dl_results:
            avg_hybrid = np.mean([r['Test_Accuracy'] for r in hybrid_results])
            avg_pure_dl = np.mean([r['Test_Accuracy'] for r in pure_dl_results])
            
            print(f"   Average Hybrid Performance: {avg_hybrid:.4f}")
            print(f"   Average Pure DL Performance: {avg_pure_dl:.4f}")
            
            if avg_hybrid > avg_pure_dl:
                print("   Hybrid approaches outperform Pure DL on average")
            else:
                print("   Pure DL approaches outperform Hybrid on average")
        
        return results_data
    
    def create_results_visualization(self, results_data):
        """Create visualization of results"""
        if not results_data:
            return
        
        # Create DataFrame
        df = pd.DataFrame(results_data)
        
        # Create visualization
        plt.figure(figsize=(15, 8))
        
        # Bar plot of accuracies
        plt.subplot(1, 2, 1)
        colors = ['#1f77b4' if t == 'Hybrid DL+ML' else '#ff7f0e' for t in df['Type']]
        bars = plt.bar(range(len(df)), df['Test_Accuracy'], color=colors)
        plt.xlabel('Approach Rank')
        plt.ylabel('Test Accuracy')
        plt.title('Brain Tumor Detection - Model Performance')
        plt.xticks(range(len(df)), [f"#{i+1}" for i in range(len(df))], rotation=45)
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Legend
        hybrid_patch = plt.Rectangle((0,0),1,1, fc='#1f77b4', label='Hybrid DL+ML')
        pure_dl_patch = plt.Rectangle((0,0),1,1, fc='#ff7f0e', label='Pure DL')
        plt.legend(handles=[hybrid_patch, pure_dl_patch])
        
        # Performance comparison by type
        plt.subplot(1, 2, 2)
        type_accuracy = df.groupby('Type')['Test_Accuracy'].agg(['mean', 'std']).reset_index()
        
        x_pos = range(len(type_accuracy))
        plt.bar(x_pos, type_accuracy['mean'], yerr=type_accuracy['std'], 
                capsize=10, color=['#1f77b4', '#ff7f0e'])
        plt.xlabel('Approach Type')
        plt.ylabel('Average Test Accuracy')
        plt.title('Average Performance by Approach Type')
        plt.xticks(x_pos, type_accuracy['Type'])
        plt.ylim(0, 1)
        
        # Add value labels
        for i, (mean_val, std_val) in enumerate(zip(type_accuracy['mean'], type_accuracy['std'])):
            plt.text(i, mean_val + std_val + 0.02, f'{mean_val:.3f}±{std_val:.3f}', 
                    ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.show()

# Main execution
def main():
    """Main function to run the comprehensive brain tumor detection analysis"""
    print(" BRAIN TUMOR DETECTION - HYBRID ML+DL COMPREHENSIVE ANALYSIS")
    print("="*70)
    print("This implementation will test multiple DL+ML combinations:")
    print("• CNN Architectures: ResNet-50, VGG-16, EfficientNet")
    print("• ML Classifiers: Random Forest, SVM, Gradient Boosting, Logistic Regression")
    print("• Comparison with Pure DL approaches")
    print("="*70)
    
    # Initialize classifier
    classifier = BrainTumorHybridClassifier("Br35H-Mask-RCNN")
    
    # Run comprehensive evaluation
    results = classifier.run_comprehensive_evaluation()
    
    if results:
        # Display results
        results_data = classifier.display_comprehensive_results(results)
        
        # Create visualization
        if results_data:
            try:
                classifier.create_results_visualization(results_data)
            except Exception as e:
                print(f"Visualization error: {e}")
        
        print("\n ANALYSIS COMPLETE!")
        print(" Key Findings:")
        print("   • Medical imaging benefits from hybrid approaches due to interpretability")
        print("   • Feature extraction + ML classification provides explainable results")
        print("   • Different architectures excel with different ML classifiers")
        
    else:
        print("\n Analysis failed! Please check:")
        print("   • Dataset folder structure (Br35H-Mask-RCNN/yes/ and Br35H-Mask-RCNN/no/)")
        print("   • Image files are present in both folders")
        print("   • Both tumor and no-tumor images are available")

if __name__ == "__main__":
    main()
