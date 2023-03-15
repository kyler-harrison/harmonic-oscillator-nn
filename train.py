import torch
import numpy as np
import pandas as pd
import time
from nn import FeedFwdNN


def main():
    df = pd.read_csv("underdamped_training_data.csv")
    feat_cols = [col for col in df.columns if col != "y"]
    X = torch.tensor(df[feat_cols].values).float()
    y = torch.tensor(df["y"].values).float()
    y = y.reshape(y.shape[0], 1)

    input_len = X.shape[1]
    net = FeedFwdNN(input_len, 1)
    num_epochs = 10
    mod_step = 100
    pred = net.forward(X[0])

    for epoch_idx in range(num_epochs):
        start = time.time()
        print(f"epoch: {epoch_idx}")

        for i in range(X.shape[0]):
            predicted_position = net.forward(X[i])
            loss = net.backward(predicted_position, y[i])

        print(f"final loss = {loss}, time = {time.time() - start:.2f} seconds")

    torch.save(net.state_dict(), "underdamped_model.pt")


if __name__ == "__main__":
    main()
