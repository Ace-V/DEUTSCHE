from aif360.datasets import GermanDataset
from aif360.metrics import ClassificationMetric
from sklearn.linear_model import LogisticRegression

dataset = GermanDataset()

X = dataset.features
y = dataset.labels.ravel()

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

dataset_pred = dataset.copy()
dataset_pred.labels = model.predict(X).reshape(-1, 1)

privileged_groups = [{"sex": 1}]
unprivileged_groups = [{"sex": 0}]

metric = ClassificationMetric(
    dataset,
    dataset_pred,
    privileged_groups=privileged_groups,
    unprivileged_groups=unprivileged_groups)

eod = metric.equal_opportunity_difference()

print("Equal Opportunity Difference:", eod)

if eod < 0:
    print("Unfair towards unprivileged group")
elif eod > 0:
    print("Unfair towards privileged group")
else:
    print("Fair towards both groups")
