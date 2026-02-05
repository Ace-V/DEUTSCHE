from aif360.datasets import GermanDataset
from aif360.metrics import BinaryLabelDatasetMetric

def disparate_impact(dataset, privileged_groups, unprivileged_group):
    
    metric= BinaryLabelDatasetMetric(dataset,
                                   unprivileged_groups=unprivileged_group,
                                    privileged_groups=privileged_groups)
    return metric.disparate_impact()

def main():
    dataset=GermanDataset()
    privileged_groups=[{'sex': 1}]
    unprivileged_group=[{'sex': 0}]

    di= disparate_impact(dataset, privileged_groups, unprivileged_group)
    print(f'Disparate Impact: {di}')

    if di < 1:
        print("Unfair towards unprivileged group")
    elif di > 1:
        print("Unfair towards privileged group")
    else:
        print("Fair towards both groups")

if __name__ == "__main__":
    main()