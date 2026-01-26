from aif360.datasets import GermanDataset
from aif360.metrics import BinaryLabelDatasetMetric

dataset= GermanDataset()

privileged_groups= [{"sex": 1}]
unprivileged_groups= [{"sex": 0}]

metric = BinaryLabelDatasetMetric(dataset,
                                  unprivileged_groups=unprivileged_groups,
                                    privileged_groups=privileged_groups)

male_apps= metric.base_rate(privileged=True)
female_apps= metric.base_rate(privileged=False)
diff= male_apps-female_apps

print(f"Male approval rate{male_apps*100:.1f}%")
print(f"Female approval rate{female_apps*100:.1f}%")
print(f"Approval rate difference: {diff*100:.1f}%")
print()

if abs(diff)<0.1:
    print("BIAS")

else:
    print("NO BIAS")