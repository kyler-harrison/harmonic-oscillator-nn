import torch 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nn import FeedFwdNN


def main():
    df = pd.read_csv("data/underdamped_range_good_test.csv")
    feat_cols = [col for col in df.columns if col != "y"]

    model = FeedFwdNN(len(feat_cols), 1)
    model.load_state_dict(torch.load("models/underdamped_independent_good_1e6.pt"))

    X = torch.tensor(df[feat_cols].values).float()
    y = df["y"].to_numpy()

    preds = model(X)
    preds = preds.detach().numpy()
    preds = preds.reshape(preds.shape[0])

    plt.plot(df["t"], y, label="analytical")
    plt.plot(df["t"], preds, "--", label="predicted")
    plt.xlabel("t")
    plt.ylabel("y")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

