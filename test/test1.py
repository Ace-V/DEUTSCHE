from aif360.datasets import GermanDataset
from aif360.metrics import ClassificationMetric

from models.logistic_regression import train_logistic_model, generate_predictions
from metrics.average_odds import average_odds_difference
from mitigation.reweighing import apply_reweighing



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
    print(f"Equal Opportunity Difference: {eod:.4f}")

    aod = average_odds_difference(
        dataset,
        dataset_pred,
        privileged_groups,
        unprivileged_groups
    )

    
    print(f"Average Odds Difference: {aod:.4f}")
    
    fpr_diff = metric.false_positive_rate_difference()
    print(f"False Positive Rate Difference: {fpr_diff:.4f}")

    tpr_diff = metric.true_positive_rate_difference()
    print(f"True Positive Rate Difference: {tpr_diff:.4f}")

    # Apply reweighing
    dataset_rw = apply_reweighing(
    dataset,
    privileged_groups,
    unprivileged_groups
    )

# Train model on reweighed data
    model_rw = train_logistic_model(dataset_rw)
    dataset_pred_rw = generate_predictions(dataset_rw, model_rw)

# Fairness evaluation after mitigation
    metric_rw = ClassificationMetric(
    dataset_rw,
    dataset_pred_rw,
    privileged_groups=privileged_groups,
    unprivileged_groups=unprivileged_groups
    )

    print("\n=== After Reweighing ===")
    print(f"EOD: {metric_rw.equal_opportunity_difference():.4f}")
    print(f"AOD: {metric_rw.average_odds_difference():.4f}")
    print(f"FPRD: {metric_rw.false_positive_rate_difference():.4f}")

    accuracy = metric_rw.accuracy()
    print(f"Accuracy after reweighing: {accuracy:.4f}")




if __name__ == "__main__":
    main()
