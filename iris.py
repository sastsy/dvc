import seaborn as sns
import matplotlib.pyplot as plt
from dvclive import Live
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris


iris = load_iris()
X = iris.data
y = iris.target


with Live() as live:
    live.log_param("epochs", 1)

    for i in range(1, 4):
        plt.clf()
        clf = DecisionTreeClassifier(max_depth=i)
        clf.fit(X, y)
        y_pred = clf.predict(X)
        live.log_metric('Precision', precision_score(y, y_pred, average='micro'))
        live.log_metric('Recall', recall_score(y, y_pred, average='micro'))
        conf_matrix = confusion_matrix(y, y_pred)
        sns_plot = sns.heatmap(conf_matrix, annot=True)
        results_path = 'results.png'
        plt.savefig(results_path)
        live.log_image(f"DecisionTree_max_depth_{i}.png", 'results.png')
        live.next_step()
        