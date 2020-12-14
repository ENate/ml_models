import matplotlib
import numpy as np
from sklearn import metrics
from sklearn.metrics import multilabel_confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support as score
import matplotlib.pyplot as plt


class ModelPredict:
    """A class to plot predictions and analyze results"""
    def __init__(self, batch_sizes, training_iterations, display_iter):
        self.int_value = None
        self.batch_sizes = batch_sizes
        self.square_value = 0.0
        self.height, self.width = None, None
        self.training_iterations = training_iterations
        self.display_iter = display_iter

    def predict_function(self, opt_params):
        """Function to compute the parameter square_value?"""
        self.int_value = opt_params
        self.square_value = self.int_value ** 2
        print('The square root value is', self.square_value)
        return self.square_value

    def plot_function(self, train_losses, train_accuracies, test_losses, test_accuracies):
        """A function to format plotting and then plot results"""
        font = {
            'family': 'Bitstream Vera Sans',
            'weight': 'bold',
            'size': 18
        }
        matplotlib.rc('font', **font)

        self.width = 12
        self.height = 12
        plt.figure(figsize=(self.width, self.height))

        indep_train_axis = np.array(range(self.batch_sizes, (len(train_losses) + 1) * self.batch_sizes, self.batch_sizes))
        plt.plot(indep_train_axis, np.array(train_losses), "b--", label="Train losses")
        plt.plot(indep_train_axis, np.array(train_accuracies), "g--", label="Train accuracies")

        indep_test_axis = np.append(
            np.array(range(self.batch_sizes, len(test_losses) * self.display_iter, self.display_iter)[:-1]),
            [self.training_iterations]
        )
        plt.plot(indep_test_axis, np.array(test_losses), "b-", label="Test losses")
        plt.plot(indep_test_axis, np.array(test_accuracies), "g-", label="Test accuracies")
        g1 = plt.figure(1)
        plt.title("Training session's progress over iterations")
        plt.legend(loc='upper right', shadow=True)
        plt.ylabel('Training Progress (Loss or Accuracy values)')
        plt.xlabel('Training iteration')
        g1.savefig('~/tasks/talpa-datascience-task/'
                   'reports/figures/training_test_progress.pdf')
        plt.show()

    def metrics_confusion_matrices(self, one_hot_predictions, accuracy, out_test_set, out_classes):
        """A function to compute confustion matrices and other statistics"""
        state_labels = ["Machine Off:4", "Idle:3", "Travelling:6", "Hole Setup:2", "Drilling:1", "Anchoring: 0",
                        "Translational Delay:5"]
        # Results
        thresh = 0.5
        # predictions = one_hot_predictions.argmax(1)
        # predictions1 = one_hot_predictions.argmax(1)  # np.argmax(one_hot_predictions, axis=1)
        # out_test_set = np.argmax(out_test_set, axis=1)
        predictions = np.argmax(one_hot_predictions, axis=1)
        one_hot_predictions = np.array([[1 if i > thresh else 0 for i in j] for j in one_hot_predictions])
        predictions = one_hot_predictions
        print(one_hot_predictions[0:4, :])
        print(out_test_set[0:4, :])
        print("Testing Accuracy: {}%".format(100 * accuracy))
        print("")
        print("Precision: {}%".format(100 * metrics.precision_score(out_test_set, one_hot_predictions, average="micro")))

        print("Recall: {}%".format(100 * metrics.recall_score(out_test_set, one_hot_predictions, average="micro")))
        print("f1_score: {}%".format(100 * metrics.f1_score(out_test_set, one_hot_predictions, average="micro")))
        print(classification_report(out_test_set, one_hot_predictions))
        print("")
        precision, recall, fscore, support = score(out_test_set, predictions)

        print('precision: {}'.format(precision))
        print('recall: {}'.format(recall))
        print('fscore: {}'.format(fscore))
        print('support: {}'.format(support))
        print("")
        print("Confusion Matrix for each label:")
        confusion_matrix2 = multilabel_confusion_matrix(out_test_set, one_hot_predictions)
        print(confusion_matrix2[0])
        out_test_set = np.argmax(out_test_set, axis=1)
        one_hot_predictions = np.argmax(one_hot_predictions, axis=1)
        print(one_hot_predictions.shape)
        confusion_matrix = metrics.confusion_matrix(out_test_set, one_hot_predictions)

        print(confusion_matrix)

        normalised_confusion_matrix = np.array(confusion_matrix, dtype=np.float32) / np.sum(confusion_matrix) * 100

        print("")
        print("Confusion matrix (normalised to % of total test data):")
        print(normalised_confusion_matrix)

        # Plot Results:
        self.width = 12
        height = 12
        g0 = plt.figure(figsize=(self.width, height))
        plt.imshow(normalised_confusion_matrix, interpolation='nearest', cmap=plt.cm.rainbow)
        # Normalized confusion matrix normalised to % of total test data
        plt.title("Normalized Confusion matrix")
        plt.colorbar()
        tick_marks = np.arange(out_classes)
        plt.xticks(tick_marks, state_labels, rotation=90)
        plt.yticks(tick_marks, state_labels)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        g0.savefig('~/tasks/talpa-datascience-task/reports/figures/confusion_matrix.pdf')
        plt.show()


if __name__ == '__main__':
    pass
