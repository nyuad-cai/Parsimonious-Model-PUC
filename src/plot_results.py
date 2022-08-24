from sklearn.metrics import accuracy_score, roc_auc_score,average_precision_score, roc_curve, f1_score,precision_recall_curve, auc
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt 
from scipy import stats
import pandas as pd
import numpy as np

 """
    https://github.com/nyuad-cai/COVID19Complications/blob/master/code/plot.py
 """

def computing_confidence_intervals(list_,true_value):
    """This function calcualts the 95% Confidence Intervals"""
    delta = (true_value - list_)
    list(np.sort(delta))
    delta_lower = np.percentile(delta, 97.5)
    delta_upper = np.percentile(delta, 2.5)

    upper = true_value - delta_upper
    lower = true_value - delta_lower
    return (upper,lower)


def plot_calibration_multi(label,true,predict, bins,name):

        """ This function plots the reliability curves based on the predictions"""

        colors=["#e60049", "#0bb4ff", "#50e991", "#e6d800", "#9b19f5", "#ffa300", "#dc0ab4", "#b3d4ff", "#00bfa0"]


        plt.plot([0,1],[0,1], 'k--')
        count = 0
        for i, j, k in zip(label, true, predict):
            fpr1, tpr1 = calibration_curve(j, k, n_bins=bins)
            text_a = label[count]
            plt.plot(tpr1, fpr1, label=text_a, color=colors[count])
            count = count +1

        ax = plt.gca()
        ax.patch.set_edgecolor('black')
        ax.patch.set_linewidth('1')
        ax.axis("on")
        plt.xlim(0,1)
        plt.ylim(0,1)
        ax.set_aspect('equal', adjustable='box')
        ax.set_facecolor('white')
        plt.xlabel("Predicted Probablity")
        plt.ylabel("True Proability in each bin")
        plt.rcParams['figure.facecolor'] = 'none'
#         plt.legend(fontsize=7)



        plt.savefig('plots/Calibration Curvefinal'+name+'.pdf', format='pdf', dpi=1400,bbox_inches="tight")
        plt.show()

def plot_roc_multi(label,true,predict,name):
        """ This function plots the ROC curves based on the predictions"""

        colors=["#e60049", "#0bb4ff", "#50e991", "#e6d800", "#9b19f5", "#ffa300", "#dc0ab4", "#b3d4ff", "#00bfa0"]

#
        plt.plot([0,1],[0,1], 'k--')
        count = 0
        for i, j, k in zip(label, true, predict):
            fpr1, tpr1,_ =  roc_curve(j, k)
            text_a = label[count]
            plt.plot(fpr1, tpr1, label=text_a ,color=colors[count])
            count = count +1

        ax = plt.gca()
        ax.patch.set_edgecolor('black')
        ax.patch.set_linewidth('1')
        ax.axis("on")
        plt.xlim(0,1)
        plt.ylim(0,1)
        ax.set_aspect('equal', adjustable='box')
        ax.set_facecolor('white')
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")

        plt.rcParams['figure.facecolor'] = 'none'
        plt.legend(fontsize=7)
        plt.savefig('plots/Roc_curve_final'+name+'.pdf', format='pdf', dpi=1400, bbox_inches="tight")

        plt.show()


def plot_PRC_multi(label,true,predict,name):
        """ This function plots the PR curves based on the predictions"""

        count = 0
        colors=["#e60049", "#0bb4ff", "#50e991", "#e6d800", "#9b19f5", "#ffa300", "#dc0ab4", "#b3d4ff", "#00bfa0"]

        for i, j, k in zip(label, true, predict):
            precision, recall,_ =  precision_recall_curve(j, k)
            text_a = label[count]
            plt.plot(recall, precision, label=text_a, color=colors[count] )
            count = count +1

        ax = plt.gca()
        ax.patch.set_edgecolor('black')
        ax.patch.set_linewidth('1')
        ax.axis("on")
        plt.xlim(0,1)
        plt.ylim(0,1)
        ax.set_aspect('equal', adjustable='box')
        ax.set_facecolor('white')
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.rcParams['figure.facecolor'] = 'none'
        plt.savefig('plots/PR_curve_final'+name+'.pdf', format='pdf', dpi=1400, bbox_inches="tight")

        plt.show()
