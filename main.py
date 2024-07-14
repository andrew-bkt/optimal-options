# main.py
from utils.config_manager import ConfigManager
from data.data_pipeline import DataPipeline
from models.model_factory import ModelFactory
from utils.results_manager import ResultsManager
from utils.logger import app_logger as logger, toggle_debug_logging

def main():
    config = ConfigManager('config.yaml')
    
    # Set debug logging based on config
    if config.get('debug_logging', False):
        toggle_debug_logging(True)
    
    data_pipeline = DataPipeline(config)
    results_manager = ResultsManager(config)

    try:
        logger.info("Starting data processing...")
        combined_data = data_pipeline.process_data()
        logger.info(f"Combined data shape: {combined_data.shape}")
        logger.info(f"Class distribution:\n{combined_data['target'].value_counts(normalize=True)}")
        
        X, y = data_pipeline.preprocess_data(combined_data)
        
        logger.info(f"Final shape of feature matrix X: {X.shape}")
        logger.info(f"Final shape of target vector y: {y.shape}")
        logger.info(f"Features used: {X.columns.tolist()}")

        results_manager = ResultsManager(config)

        for model_config in config.get('models'):
            logger.info(f"Training and cross-validating {model_config['name']} model...")
            model = ModelFactory.create_model(model_config['type'], X, y, model_config['params'])
            model.train()
            results_manager.save_results(model_config['name'], model, X, y)

        results_manager.plot_model_comparison()
        results_manager.print_summary()

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.error("Please check the error message and your data processing steps.")

if __name__ == "__main__":
    main()
