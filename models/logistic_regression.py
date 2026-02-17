from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def train_logistic_model(dataset):


    X = dataset.features
    y = dataset.labels.ravel()

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    model.fit(X, y)

    return model


def generate_predictions(dataset, model):
    """
    Generates predicted labels and returns a new AIF360 dataset
    with predictions attached.
    """

    X = dataset.features

    predicted_labels = model.predict(X)

    dataset_pred = dataset.copy()
    dataset_pred.labels = predicted_labels.reshape(-1, 1)

    return dataset_pred
