import torch
import numpy as np
import pandas as pd
import time
from nn import FeedFwdNN


def main():
    # load and prepare train dataset
    df = pd.read_csv("data/underdamped_independent_train.csv")
    feat_cols = [col for col in df.columns if col != "y"]
    X = torch.tensor(df[feat_cols].values).float()
    y = torch.tensor(df["y"].values).float()
    y = y.reshape(y.shape[0], 1)

    # load and prepare test dataset
    test_df = pd.read_csv("data/underdamped_independent_test.csv")
    X_test = torch.tensor(test_df[feat_cols].values).float()
    y_test = torch.tensor(test_df["y"].values).float()
    y_test = y_test.reshape(y_test.shape[0], 1)

    # initialize network
    input_len = X.shape[1]
    num_outputs = 1  # nn predicts a scalar
    model = FeedFwdNN(input_len, num_outputs)

    # parameters of training loop
    num_epochs = 10
    batch_size = 10
    batches = torch.arange(0, X.shape[0], batch_size)
    
    # training loop
    for epoch_idx in range(num_epochs):
        start = time.time()
        print(f"epoch: {epoch_idx}")

        for batch in batches:
            X_batch = X[batch:batch + batch_size]
            y_batch = y[batch:batch + batch_size]
            y_predictions = model.forward(X_batch)
            train_loss = model.backward(y_predictions, y_batch)

        y_test_predictions = model.forward(X_test)
        test_loss = model.compute_loss(y_test_predictions, y_test)
        print(f"train loss = {train_loss}, test loss = {test_loss}, time = {time.time() - start:.2f} seconds")

    torch.save(model.state_dict(), "models/underdamped_good_1e6_t0=0_tf=50_ts=0.1.pt")


if __name__ == "__main__":
    main()
