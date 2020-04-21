import glob
import pandas as pd

def read_metrics():
    path = glob.glob("*.csv")
    epochs = [1,2,3,4,5]
    for dataset_path in path:
        dataset = pd.read_csv(dataset_path, sep=';')
        dataset = dataset.round(3)
        dataset['epoch'] = epochs
        dataset.to_csv(dataset_path, sep=';')

read_metrics()

