import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.projections")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix, 
    f1_score, 
    precision_score, 
    recall_score, 
)


def plot_loss(train_loss, val_loss):

    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, train_loss, label="Training loss")
    plt.plot(epochs, val_loss, label="Validation loss")

    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.title("Training and Validation Loss", fontsize=16)

    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.savefig("results/loss_curves.png", dpi=300, bbox_inches="tight")
    plt.close()


def get_performances(ytrue, yhat, classes, plot_confusion=False):

    acc_unbal = accuracy_score(ytrue, yhat)
    acc_weigh = balanced_accuracy_score(ytrue, yhat)
    f1_weigh = f1_score(ytrue, yhat, average = 'weighted', zero_division = 0.0)
    prec_weigh = precision_score(ytrue, yhat, average = 'weighted',zero_division = 0.0)
    recall_weigh = recall_score(ytrue, yhat, average = 'weighted', zero_division = 0.0)

    print(' ')
    print('           |-----------------------------------------|')
    print('           |              SCORE SUMMARY              |')
    print('           |-----------------------------------------|')
    print('           |  Accuracy score:                 %.3f  |' %acc_unbal) 
    print('           |  Accuracy score weighted:        %.3f  |' %acc_weigh) 
    print('           |-----------------------------------------|')
    print('           |  Precision score weighted:       %.3f  |' %prec_weigh)
    print('           |-----------------------------------------|')
    print('           |  Recall score weighted:          %.3f  |' %recall_weigh)
    print('           |-----------------------------------------|')
    print('           |  F1-score weighted:              %.3f  |' %f1_weigh)
    print('           |-----------------------------------------|')
    print(' ')

    if plot_confusion:

        # Compute the confmats
        classes_str = [str(i) for i in classes]
        ConfMat = confusion_matrix(ytrue, yhat, labels=classes).T
        ConfMat_df = pd.DataFrame(ConfMat, index = classes_str, columns = classes_str)
        Acc_mat = confusion_matrix(ytrue, yhat, labels=classes, normalize='true').T
        Acc_mat_df = pd.DataFrame(Acc_mat, index = classes_str, columns = classes_str)

        # Helper function for confmats
        def plot_confmat(
            ax,
            mat_df,
            diag_mask,
            cmap_off="Blues",
            cmap_diag="OrRd",
            vmin=None,
            vmax=None,
            fmt="d",
            title="",
            annot_size=12,
            show_ylabel=True,
        ):
            # Off-diagonal
            sns.heatmap(
                mat_df,
                mask=~diag_mask,
                annot=True,
                fmt=fmt,
                cmap=cmap_off,
                vmin=vmin,
                vmax=vmax,
                linewidths=1,
                cbar=False,
                annot_kws={"size": annot_size},
                ax=ax
            )

            # Diagonal
            sns.heatmap(
                mat_df,
                mask=diag_mask,
                annot=True,
                fmt=fmt,
                cmap=cmap_diag,
                vmin=vmin,
                vmax=vmax,
                linewidths=1,
                cbar=False,
                annot_kws={"size": annot_size},
                ax=ax
            )

            ax.set_title(title, fontsize=20)
            ax.set_xlabel("True labels", fontsize=16)

            if show_ylabel:
                ax.set_ylabel("Predicted labels", fontsize=16)
            else:
                ax.set_ylabel("")

            ax.invert_yaxis()

        # General settings
        font_size = 12
        off_diag_mask = np.eye(*ConfMat.shape, dtype=bool)
        sns.set(font_scale=1.5)

        # Raw Confmat
        plt.figure(figsize=(7, 6), layout="constrained")

        plot_confmat(
            ax=plt.gca(),
            mat_df=ConfMat_df,
            diag_mask=off_diag_mask,
            vmin=ConfMat.min(),
            vmax=ConfMat.max(),
            fmt="d",
            annot_size=font_size,
            title="Confusion Matrix"
        )

        plt.savefig("results/confusion_matrix_raw.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Normalized Confmat
        plt.figure(figsize=(7, 6), layout="constrained")

        plot_confmat(
            ax=plt.gca(),
            mat_df=Acc_mat_df,
            diag_mask=off_diag_mask,
            vmin=0.0,
            vmax=1.0,
            fmt=".2f",
            annot_size=font_size,
            title="Normalized Confusion Matrix"
        )

        plt.savefig("results/confusion_matrix_normalized.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Joint Confmats
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout="constrained")

        # Raw
        plot_confmat(
            ax=axes[0],
            mat_df=ConfMat_df,
            diag_mask=off_diag_mask,
            vmin=ConfMat.min(),
            vmax=ConfMat.max(),
            fmt="d",
            title="Confusion Matrix",
            show_ylabel=True
        )

        # Normalized
        plot_confmat(
            ax=axes[1],
            mat_df=Acc_mat_df,
            diag_mask=off_diag_mask,
            vmin=0.0,
            vmax=1.0,
            fmt=".2f",
            title="Normalized Confusion Matrix",
            show_ylabel=False
        )

        plt.savefig(
            "results/confusion_matrix_combined.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()