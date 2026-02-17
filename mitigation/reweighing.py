from aif360.algorithms.preprocessing import Reweighing


def apply_reweighing(dataset, privileged_groups, unprivileged_groups):
    """
    Applies Reweighing algorithm to dataset.
    
    Returns:
    - transformed dataset
    """

    rw = Reweighing(
        privileged_groups=privileged_groups,
        unprivileged_groups=unprivileged_groups
    )

    rw.fit(dataset)
    transformed_dataset = rw.transform(dataset)

    return transformed_dataset
