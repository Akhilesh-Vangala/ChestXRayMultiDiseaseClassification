import argparse
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf

sys.path.append(str(Path(__file__).parent))

from data_preprocessing import NIHChestXRayPreprocessor
from training import ModelTrainer
from evaluation import ModelEvaluator
from gradcam import GradCAMVisualizer
from utils import set_seed, setup_logging, save_config

DISEASE_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
    'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
    'Pleural_Thickening', 'Hernia'
]


def main():
    parser = argparse.ArgumentParser(description='Chest X-Ray Multi-Disease Classification')
    parser.add_argument('--data_path', type=str, default='data/raw', help='Path to NIH Chest X-Ray data')
    parser.add_argument('--images_dir', type=str, default='data/raw/images', help='Path to images directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--use_focal_loss', action='store_true', default=True, help='Use focal loss')
    parser.add_argument('--use_weighted_sampling', action='store_true', default=True, help='Use weighted sampling')
    parser.add_argument('--use_ensemble', action='store_true', help='Train ensemble of models')
    parser.add_argument('--use_tta', action='store_true', help='Use test-time augmentation')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    
    args = parser.parse_args()
    
    set_seed(42)
    logger = setup_logging(Path(args.output_dir) / 'logs')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Chest X-Ray Multi-Disease Classification")
    print("=" * 70)
    
    print("\n[1/6] Data Preprocessing...")
    preprocessor = NIHChestXRayPreprocessor(data_path=args.data_path)
    data_splits = preprocessor.load_data_splits()
    
    print(f"Training samples: {len(data_splits['train']):,}")
    print(f"Validation samples: {len(data_splits['val']):,}")
    print(f"Test samples: {len(data_splits['test']):,}")
    
    print("\n[2/6] Class Imbalance Analysis...")
    imbalance_df = preprocessor.analyze_class_imbalance(data_splits['train'])
    print(imbalance_df.to_string(index=False))
    print(f"Maximum class imbalance ratio: {imbalance_df['Ratio'].max():.2f}:1")
    
    class_weights = preprocessor.get_class_weights(data_splits['train'])
    
    print("\n[3/6] Creating Datasets...")
    train_dataset = preprocessor.create_dataset(
        data_splits['train'], args.images_dir, batch_size=args.batch_size,
        shuffle=True, weighted=args.use_weighted_sampling,
        use_augmentation=True, use_mixup=True, use_cutout=True, use_cutmix=True
    )
    val_dataset = preprocessor.create_dataset(
        data_splits['val'], args.images_dir, batch_size=args.batch_size,
        shuffle=False, weighted=False
    )
    test_dataset = preprocessor.create_dataset(
        data_splits['test'], args.images_dir, batch_size=args.batch_size,
        shuffle=False, weighted=False
    )
    
    print("\n[4/6] Model Training...")
    trainer = ModelTrainer(model_dir=output_dir / 'models')
    base_models = ['ResNet50', 'DenseNet121', 'EfficientNetB0'] if args.use_ensemble else ['ResNet50']
    
    training_results = trainer.train(
        train_dataset, val_dataset, epochs=args.epochs,
        use_focal_loss=args.use_focal_loss, class_weights=class_weights,
        learning_rate=args.learning_rate, base_models=base_models,
        use_mixed_precision=True, use_lookahead=True
    )
    
    model = training_results['ensemble'] if training_results['ensemble'] else training_results['models'][0]
    print(f"Model parameters: {model.count_params():,}")
    print(f"Best model saved to: {training_results['best_model_path']}")
    
    print("\n[5/6] Model Evaluation...")
    evaluator = ModelEvaluator(DISEASE_LABELS)
    test_results = evaluator.evaluate(model, test_dataset, use_tta=args.use_tta)
    
    print(f"\nTest Set Performance:")
    print(f"Mean AUC-ROC: {test_results['mean_auc']:.4f}")
    print(f"Mean Average Precision: {test_results['mean_ap']:.4f}")
    print(f"\nPer-Class Performance:")
    print(test_results['results_df'].to_string(index=False))
    
    test_results['results_df'].to_csv(output_dir / 'metrics' / 'test_performance.csv', index=False)
    
    print("\n[6/6] GradCAM Visualization...")
    gradcam = GradCAMVisualizer(model, DISEASE_LABELS, output_dir=output_dir / 'gradcam')
    print("GradCAM visualizer initialized")
    
    config = {
        'data_path': args.data_path, 'images_dir': args.images_dir,
        'epochs': args.epochs, 'batch_size': args.batch_size,
        'learning_rate': args.learning_rate, 'use_focal_loss': args.use_focal_loss,
        'use_weighted_sampling': args.use_weighted_sampling,
        'use_ensemble': args.use_ensemble, 'use_tta': args.use_tta,
        'mean_auc': float(test_results['mean_auc']), 'mean_ap': float(test_results['mean_ap'])
    }
    
    save_config(config, output_dir / 'training_config.json')
    
    print("\n" + "=" * 70)
    print("Training and Evaluation Complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
