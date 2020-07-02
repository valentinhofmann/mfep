import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def main():

    # Load data
    df = pd.read_pickle('../../data/final/data_new.p')

    # Initialize lists to store prediction scores and feature importances for delta_a models
    scores = []
    imps = []

    # Loop over different thresholds
    for theta in np.arange(0, 3.9, 0.8):

        df_temp = df.copy()

        # Define growing families
        df_temp['y_true'] = df_temp.delta_a > theta
        df_temp.y_true = df_temp.y_true.astype(int)

        # Create balanced dataset
        g = df_temp.groupby('y_true')
        df_temp = g.apply(lambda x: x.sample(g.size().min(), random_state=0).reset_index(drop=True))
        df_temp.index = df_temp.index.droplevel()

        # Split into train, development, and test data
        train_dev, test = train_test_split(df_temp, test_size=0.2, random_state=0, stratify=df_temp[['y_true']])
        train, dev = train_test_split(train_dev, test_size=0.2, random_state=0, stratify=train_dev[['y_true']])
        X_train = train[['F', 'f_r', 'z^p', 'D^u', 'D^t', 'Q']]
        y_train = train[['y_true']]
        X_test = test[['F', 'f_r', 'z^p', 'D^u', 'D^t', 'Q']]
        y_test = test[['y_true']]

        # Train Random Forest model
        rf_model = RandomForestClassifier(n_estimators=80, max_depth=20, random_state=0)
        rf_model.fit(X_train, y_train.values.ravel())

        # Store prediction scores and feature importances
        scores.append(rf_model.score(X_test, y_test))
        imps.append(rf_model.feature_importances_)

    # Write results to text files
    with open('results/scores_new_delta_a.txt', 'w') as f:
        f.write(' '.join(['{:3.3f}'.format(score) for score in scores]))
        f.write(' {:3.3f} $\\pm$ {:3.3f} '.format(np.mean(scores), np.std(scores)))

    # Initialize lists to store prediction scores and feature importances for delta_r models
    scores = []
    imps = []

    # Loop over different thresholds
    for theta in np.arange(1, 2.2, 0.2):

        df_temp = df.copy()

        # Define growing families
        df_temp['y_true'] = df_temp.delta_r > theta
        df_temp.y_true = df_temp.y_true.astype(int)

        # Create balanced dataset
        g = df_temp.groupby('y_true')
        df_temp = g.apply(lambda x: x.sample(g.size().min(), random_state=0).reset_index(drop=True))
        df_temp.index = df_temp.index.droplevel()

        # Split into train, development, and test data
        train_dev, test = train_test_split(df_temp, test_size=0.2, random_state=0, stratify=df_temp[['y_true']])
        train, dev = train_test_split(train_dev, test_size=0.2, random_state=0, stratify=train_dev[['y_true']])
        X_train = train[['F', 'f_r', 'z^p', 'D^u', 'D^t', 'Q']]
        y_train = train[['y_true']]
        X_test = test[['F', 'f_r', 'z^p', 'D^u', 'D^t', 'Q']]
        y_test = test[['y_true']]

        # Train Random Forest model
        rf_model = RandomForestClassifier(n_estimators=80, max_depth=20, random_state=0)
        rf_model.fit(X_train, y_train.values.ravel())

        # Store prediction scores and feature importances
        scores.append(rf_model.score(X_test, y_test))
        imps.append(rf_model.feature_importances_)

    # Write results to text files
    with open('results/scores_new_delta_r.txt', 'w') as f:
        f.write(' '.join(['{:3.3f}'.format(score) for score in scores]))
        f.write(' {:3.3f} $\\pm$ {:3.3f} '.format(np.mean(scores), np.std(scores)))


if __name__ == '__main__':
    main()
