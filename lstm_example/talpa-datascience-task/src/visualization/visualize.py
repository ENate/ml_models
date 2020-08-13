# ## PLOTTING UTILITY FUNCTIONS ###
import sys
sys.path.append('../data')
import matplotlib.pyplot as plt
from make_dataset import make_data_set
import seaborn as sns

cm_lr=[[4990, 0], [2,    0]]

cm2 = [[3808,   26], [17, 1141]]

cm3 = [[4782,   28], [16,  166]]

cm4 = [[4546,  20], [20,  406]]

cm5 = [[3066,    7], [9, 1910]]

cm6 = [[4333,   20], [44,  595]]


# ## Plot activity for training set indicated by bars ###

def activity_plot(raw_data_set_with_activity):
    """
    :param raw_data_set_with_activity: formatted data set containing activity column
    :return:
    """
    g3 = plt.Figure()
    raw_data_set_with_activity['activity'].value_counts().plot(kind='bar', title='Training examples by activity type')
    g3.savefig('/home/nath/tasks/talpa-datascience-task/reports/figures/activity_plots.pdf')


# Plot some confusion matrices
g1 = plt.figure(figsize=(20, 12))

g1.suptitle("Confusion Matrices for different Classes", fontsize=24)
plt.subplots_adjust(wspace = 0.4, hspace= 0.4)

plt.subplot(2,3,1)
plt.title("Anchoring")
ax = sns.heatmap(cm_lr, annot=True, cmap="Blues",fmt="d", cbar=False, annot_kws={"size": 24})
ax.set_ylim(2.0, 0)

plt.subplot(2,3,2)
plt.title("Drilling")
ax1 = sns.heatmap(cm2, annot=True, cmap="Blues", fmt="d", cbar=False, annot_kws={"size": 24})
ax1.set_ylim(2.0, 0)

plt.subplot(2, 3, 3)
plt.title("Hole Setup")
ax2 = sns.heatmap(cm3,annot=True, cmap="Blues", fmt="d", cbar=False, annot_kws={"size": 24})
ax2.set_ylim(2.0, 0)

plt.subplot(2, 3, 4)
plt.title("Idle ")
ax3=sns.heatmap(cm4,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})
ax3.set_ylim(2.0, 0)

plt.subplot(2,3,5)
plt.title("Machine Off")
ax4=sns.heatmap(cm5, annot=True, cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})
ax4.set_ylim(2.0, 0)

plt.subplot(2,3,6)
plt.title("Translational Delay")
ax5=sns.heatmap(cm6,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})
# check current axes using:
# ax.get_ylim()
ax5.set_ylim(2.0, 0)
g1.savefig('/home/nath/tasks/talpa-datascience-task/reports/figures/confusion_matrix_plots.pdf')
plt.show()


if __name__ == '__main__':
    original_data_set = '~/tf-codes-implementation/talpa-datascience-task/data/raw/data_case_study.csv'
    data_contain_activity, data_no_activity, data_label_cols = make_data_set(original_data_set)
    activity_plot(data_contain_activity)
