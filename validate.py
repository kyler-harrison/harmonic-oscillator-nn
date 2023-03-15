import torch 
import numpy as np
import pandas as pd
from nn import FeedFwdNN
from underdamped import plot_data


def main():
    good_df = pd.read_csv("underdamped_good_test_data.csv")
    quick_df = pd.read_csv("underdamped_quick_damp_test_data.csv")
    feat_cols = [col for col in good_df.columns if col != "y"]

    model = FeedFwdNN(len(feat_cols), 1)
    model.load_state_dict(torch.load("underdamped_model.pt"))

    good_X = torch.tensor(good_df[feat_cols].values).float()
    good_y = good_df["y"].to_numpy()

    quick_X = torch.tensor(quick_df[feat_cols].values).float()
    quick_y = quick_df["y"].to_numpy()

    good_preds = model(good_X)
    good_preds = good_preds.detach().numpy()
    good_preds = good_preds.reshape(good_preds.shape[0])

    quick_preds = model(quick_X)
    quick_preds = quick_preds.detach().numpy()
    quick_preds = quick_preds.reshape(quick_preds.shape[0])

    plot_data(good_df["t"], good_preds)
    plot_data(good_df["t"], good_y)

    plot_data(quick_df["t"], quick_preds)
    plot_data(quick_df["t"], quick_y)


if __name__ == "__main__":
    main()

