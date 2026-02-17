from aif360.datasets import GermanDataset
from aif360.metrics import ClassificationMetric

from models.logistic_regression import train_logistic_model, generate_predictions


def main():
    dataset = GermanDataset()

    model = train_logistic_model(dataset)

    dataset_pred = generate_predictions(dataset, model)

    privileged_groups = [{"sex": 1}]
    unprivileged_groups = [{"sex": 0}]

    metric = ClassificationMetric(
        dataset,
        dataset_pred,
        privileged_groups=privileged_groups,
        unprivileged_groups=unprivileged_groups
    )

    eod = metric.equal_opportunity_difference()

    print("=== Model Fairness Evaluation ===")
    print(f"Equal Opportunity Difference: {eod:.4f}")


if __name__ == "__main__":
    main()
