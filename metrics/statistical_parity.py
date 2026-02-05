from aif360.datasets import GermanDataset
from aif360.metrics import BinaryLabelDatasetMetric

def statistical_parity_difference(dataset, privileged_groups, unprivileged_groups):

    metric=BinaryLabelDatasetMetric(dataset,
                                   unprivileged_groups=unprivileged_groups,
                                      privileged_groups=privileged_groups)
    
    return metric.statistical_parity_difference()

def main():
    dataset=GermanDataset()
    privileged_groups=[{'sex': 1}]
    unprivileged_groups=[{'sex': 0}]

    spd=statistical_parity_difference(dataset, privileged_groups, unprivileged_groups)
    print(f'Statistical Parity Difference: {spd}')

    if spd< -0.05:
        print("Unfair towards unprivileged group")
    elif spd> 0.05:
        print("Unfair towards privileged group")
    else:
        print("Fair towards both groups")

if __name__ == "__main__":
    main()