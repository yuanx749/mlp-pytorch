import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_learning_curve(input_path, title, output_path):
    df = pd.read_csv(input_path)
    fig, ax1 = plt.subplots()
    ax1.plot(df['loss'])
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.grid()
    ax2 = ax1.twinx()
    ax2.plot(df['train_score'], color='orange', label='training score')
    ax2.plot(df['val_score'], color='indigo', label='validation score')
    ax2.set_ylabel('Score')
    ax2.legend(loc='center right')
    ax1.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path)
    return