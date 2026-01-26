from aif360.datasets import GermanDataset
from aif360.metrics import BinaryLabelDatasetMetric
import matplotlib.pyplot as plt

dataset= GermanDataset()

metric = BinaryLabelDatasetMetric(dataset,
                                  unprivileged_groups=[{"sex": 0}],
                                  privileged_groups=[{"sex": 1}])

male_apps= metric.base_rate(privileged=True)*100
female_apps= metric.base_rate(privileged=False)*100

fig , ax = plt.subplots(figsize=(8, 6))

groups= ["Male Applicants", "Female Applicants"]
app_rates= [male_apps,female_apps]
colors=['#0018A8',"#00AEEF"] #DEUTSCHE COLOR

bars= ax.bar(groups, app_rates, color=colors,width=0.4)

ax.axhline(y=65, color='red', linestyle='--', label='Fairness')
ax.set_ylabel('Approval Rate (%)')
ax.set_title('Loan Approval Rates by Gender')
ax.set_ylim(0, 100)

for bar , app_rates in zip(bars,app_rates):
    height= bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, height, f'{app_rates:.1f}%', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('deutsche_first_chart.png',dpi=870)
print("CHART")
plt.show()