import os
import argparse
import logging
import time
import sys
import pandas as pd
import yaml

import data_loader
import nn
import plotting

parser = argparse.ArgumentParser(description='Run an experiment')
parser.add_argument('--output', default='output', type=str, help='Output directory')
parser.add_argument('--dataset', default='biodeg.csv', type=str, help='Input dataset file')
parser.add_argument('--model', default='MLP', choices=['MLP', 'CNN', 'RNN'], help='Model type')
parser.add_argument('--params', default='params.yml', type=str, help='Hyperparameters YAML')
parser.add_argument('--plot', action='store_false', help='NOT plot learning curves')
parser.add_argument('--verbose', action='store_false', help='NOT print progress messages to stdout')
args = parser.parse_args()

log_dir = os.path.abspath('log')
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, time.strftime('%Y%m%d-%H%M%S.log'))
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler(stream=sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
if not args.verbose:
    console_handler.setLevel(logging.WARNING)

def main():
    input_path = os.path.abspath(os.path.join('./data', args.dataset))
    dataset = os.path.splitext(args.dataset)[0]
    logger.info('Load {}'.format(input_path))
    params = {'test_size': 0.2, 'random_state': 1, 'cluster': 'kmeans'}
    X_train, X_test, y_train, y_test = data_loader.load(input_path, **params)
    logger.info('Split into train and test subsets: {}'.format(params))
    
    params_path = os.path.abspath(os.path.join('./params', args.params))
    with open(params_path) as file_:
        params = yaml.load(file_, Loader=yaml.SafeLoader)
    logger.info('Load {}'.format(params_path))
    logger.info('Hyperparameters: {}'.format(params))
    models = {'MLP': nn.MLPClassifier, 'CNN': nn.CNNClassifier, 'RNN': nn.RNNClassifier}
    clf = models[args.model](**params)
    estimator = clf.__class__.__name__
    logger.info('Train {} on {}'.format(estimator, dataset))
    clf.fit(X_train, y_train)
    
    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)
    csv_log = pd.DataFrame({'loss': clf.loss_curve_, 'train_score': clf.training_scores_, 'val_score': clf.validation_scores_})
    csv_log_path = os.path.join(output_dir, time.strftime('%Y%m%d-%H%M%S.csv'))
    csv_log.to_csv(csv_log_path)
    logger.info('Save learning log to {}'.format(csv_log_path))

    if args.plot:
        plot_path = os.path.join(output_dir, time.strftime('%Y%m%d-%H%M%S.png'))
        plotting.plot_learning_curve(csv_log_path, '{} on {}'.format(estimator, dataset), plot_path)
        logger.info('Save learning curves to {}'.format(plot_path))
    
    logger.info('Training score: {}'.format(clf.score(X_train, y_train)))
    logger.info('Testing score: {}'.format(clf.score(X_test, y_test)))
    logger.info('Done')

if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        logger.error(err, exc_info=True)
        raise