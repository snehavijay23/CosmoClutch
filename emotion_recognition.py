from data_extractor import load_data
from utils import extract_feature, AVAILABLE_EMOTIONS
from create_csv import write_emodb_csv, write_tess_ravdess_csv, write_custom_csv

from sklearn.metrics import accuracy_score, make_scorer, fbeta_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as pl
from time import time
from utils import get_best_estimators, get_audio_config
import numpy as np
import tqdm
import os
import random
import pandas as pd


class EmotionRecognizer:

    def __init__(self, model=None, **kwargs):
        self.emotions = kwargs.get("emotions", ["sad", "neutral", "happy"])
        self._verify_emotions()
        self.features = kwargs.get("features", ["mfcc", "chroma", "mel"])
        self.audio_config = get_audio_config(self.features)
        self.tess_ravdess = kwargs.get("tess_ravdess", True)
        self.emodb = kwargs.get("emodb", True)
        self.custom_db = kwargs.get("custom_db", True)

        if not self.tess_ravdess and not self.emodb and not self.custom_db:
            self.tess_ravdess = True

        self.classification = kwargs.get("classification", True)
        self.balance = kwargs.get("balance", True)
        self.override_csv = kwargs.get("override_csv", True)
        self.verbose = kwargs.get("verbose", 1)

        self.tess_ravdess_name = kwargs.get("tess_ravdess_name",
                                            "tess_ravdess.csv")
        self.emodb_name = kwargs.get("emodb_name", "emodb.csv")
        self.custom_db_name = kwargs.get("custom_db_name", "custom.csv")

        self.verbose = kwargs.get("verbose", 1)

        self._set_metadata_filenames()
        self.write_csv()
        self.data_loaded = False
        self.model_trained = False

        if not model:
            self.determine_best_model()
        else:
            self.model = model

    def _set_metadata_filenames(self):

        train_desc_files, test_desc_files = [], []
        if self.tess_ravdess:
            train_desc_files.append(f"train_{self.tess_ravdess_name}")
            test_desc_files.append(f"test_{self.tess_ravdess_name}")
        if self.emodb:
            train_desc_files.append(f"train_{self.emodb_name}")
            test_desc_files.append(f"test_{self.emodb_name}")
        if self.custom_db:
            train_desc_files.append(f"train_{self.custom_db_name}")
            test_desc_files.append(f"test_{self.custom_db_name}")

        # set them to be object attributes
        self.train_desc_files = train_desc_files
        self.test_desc_files = test_desc_files

    def _verify_emotions(self):

        for emotion in self.emotions:
            assert emotion in AVAILABLE_EMOTIONS, "Emotion not recognized."

    def get_best_estimators(self):
        return get_best_estimators(self.classification)

    def write_csv(self):

        for train_csv_file, test_csv_file in zip(self.train_desc_files,
                                                 self.test_desc_files):
            # not safe approach
            if os.path.isfile(train_csv_file) and os.path.isfile(test_csv_file):
                # file already exists, just skip writing csv files
                if not self.override_csv:
                    continue
            if self.emodb_name in train_csv_file:
                write_emodb_csv(self.emotions,
                                train_name=train_csv_file,
                                test_name=test_csv_file,
                                verbose=self.verbose)
                if self.verbose:
                    print("[+] Generated EMO-DB CSV File")
            elif self.tess_ravdess_name in train_csv_file:
                write_tess_ravdess_csv(self.emotions,
                                       train_name=train_csv_file,
                                       test_name=test_csv_file,
                                       verbose=self.verbose)
                if self.verbose:
                    print("[+] Generated TESS & RAVDESS DB CSV File")
            elif self.custom_db_name in train_csv_file:
                write_custom_csv(emotions=self.emotions,
                                 train_name=train_csv_file,
                                 test_name=test_csv_file,
                                 verbose=self.verbose)
                if self.verbose:
                    print("[+] Generated Custom DB CSV File")

    def load_data(self):

        if not self.data_loaded:
            result = load_data(self.train_desc_files,
                               self.test_desc_files,
                               self.audio_config,
                               self.classification,
                               emotions=self.emotions,
                               balance=self.balance)
            self.X_train = result['X_train']
            self.X_test = result['X_test']
            self.y_train = result['y_train']
            self.y_test = result['y_test']
            self.train_audio_paths = result['train_audio_paths']
            self.test_audio_paths = result['test_audio_paths']
            self.balance = result["balance"]
            if self.verbose:
                print("[+] Data loaded")
            self.data_loaded = True

    def train(self, verbose=1):

        if not self.data_loaded:
            # if data isn't loaded yet, load it then
            self.load_data()
        if not self.model_trained:
            self.model.fit(X=self.X_train, y=self.y_train)
            self.model_trained = True
            if verbose:
                print("[+] Model trained")

    def predict(self, audio_path):

        feature = extract_feature(audio_path,
                                  **self.audio_config).reshape(1, -1)
        return self.model.predict(feature)[0]

    def predict_proba(self, audio_path):

        if self.classification:
            feature = extract_feature(audio_path,
                                      **self.audio_config).reshape(1, -1)
            proba = self.model.predict_proba(feature)[0]
            result = {}
            for emotion, prob in zip(self.model.classes_, proba):
                result[emotion] = prob
            return result
        else:
            raise NotImplementedError(
                "Probability prediction doesn't make sense for regression")

    def grid_search(self, params, n_jobs=2, verbose=1):

        score = accuracy_score if self.classification else mean_absolute_error
        grid = GridSearchCV(estimator=self.model,
                            param_grid=params,
                            scoring=make_scorer(score),
                            n_jobs=n_jobs,
                            verbose=verbose,
                            cv=3)
        grid_result = grid.fit(self.X_train, self.y_train)
        return grid_result.best_estimator_, grid_result.best_params_, grid_result.best_score_

    def determine_best_model(self):

        if not self.data_loaded:
            self.load_data()

        estimators = self.get_best_estimators()

        result = []

        if self.verbose:
            estimators = tqdm.tqdm(estimators)

        for estimator, params, cv_score in estimators:
            if self.verbose:
                estimators.set_description(
                    f"Evaluating {estimator.__class__.__name__}")
            detector = EmotionRecognizer(estimator,
                                         emotions=self.emotions,
                                         tess_ravdess=self.tess_ravdess,
                                         emodb=self.emodb,
                                         custom_db=self.custom_db,
                                         classification=self.classification,
                                         features=self.features,
                                         balance=self.balance,
                                         override_csv=False)
            detector.X_train = self.X_train
            detector.X_test = self.X_test
            detector.y_train = self.y_train
            detector.y_test = self.y_test
            detector.data_loaded = True
            detector.train(verbose=0)
            accuracy = detector.test_score()
            result.append((detector.model, accuracy))

        result = sorted(result,
                        key=lambda item: item[1],
                        reverse=self.classification)
        best_estimator = result[0][0]
        accuracy = result[0][1]
        self.model = best_estimator
        self.model_trained = True
        if self.verbose:
            if self.classification:
                print(
                    f"[+] Best model determined: {self.model.__class__.__name__} with {accuracy*100:.3f}% test accuracy"
                )
            else:
                print(
                    f"[+] Best model determined: {self.model.__class__.__name__} with {accuracy:.5f} mean absolute error"
                )

    def test_score(self):
        y_pred = self.model.predict(self.X_test)
        if self.classification:
            return accuracy_score(y_true=self.y_test, y_pred=y_pred)
        else:
            return mean_squared_error(y_true=self.y_test, y_pred=y_pred)

    def train_score(self):
        y_pred = self.model.predict(self.X_train)
        if self.classification:
            return accuracy_score(y_true=self.y_train, y_pred=y_pred)
        else:
            return mean_squared_error(y_true=self.y_train, y_pred=y_pred)

    def train_fbeta_score(self, beta):
        y_pred = self.model.predict(self.X_train)
        return fbeta_score(self.y_train, y_pred, beta, average='micro')

    def test_fbeta_score(self, beta):
        y_pred = self.model.predict(self.X_test)
        return fbeta_score(self.y_test, y_pred, beta, average='micro')

    def confusion_matrix(self, percentage=True, labeled=True):
        if not self.classification:
            raise NotImplementedError(
                "Confusion matrix works only when it is a classification problem"
            )
        y_pred = self.model.predict(self.X_test)
        matrix = confusion_matrix(self.y_test, y_pred,
                                  labels=self.emotions).astype(np.float32)
        if percentage:
            for i in range(len(matrix)):
                matrix[i] = matrix[i] / np.sum(matrix[i])
            # make it percentage
            matrix *= 100
        if labeled:
            matrix = pd.DataFrame(
                matrix,
                index=[f"true_{e}" for e in self.emotions],
                columns=[f"predicted_{e}" for e in self.emotions])
        return matrix

    def draw_confusion_matrix(self):
        matrix = self.confusion_matrix(percentage=False, labeled=False)
        #TODO: add labels, title, legends, etc.
        pl.imshow(matrix, cmap="binary")
        pl.show()

    def get_n_samples(self, emotion, partition):
        if partition == "test":
            return len([y for y in self.y_test if y == emotion])
        elif partition == "train":
            return len([y for y in self.y_train if y == emotion])

    def get_samples_by_class(self):
        if not self.data_loaded:
            self.load_data()
        train_samples = []
        test_samples = []
        total = []
        for emotion in self.emotions:
            n_train = self.get_n_samples(emotion, "train")
            n_test = self.get_n_samples(emotion, "test")
            train_samples.append(n_train)
            test_samples.append(n_test)
            total.append(n_train + n_test)

        total.append(sum(train_samples) + sum(test_samples))
        train_samples.append(sum(train_samples))
        test_samples.append(sum(test_samples))
        return pd.DataFrame(data={
            "train": train_samples,
            "test": test_samples,
            "total": total
        },
                            index=self.emotions + ["total"])

    def get_random_emotion(self, emotion, partition="train"):
        if partition == "train":
            index = random.choice(list(range(len(self.y_train))))
            while self.y_train[index] != emotion:
                index = random.choice(list(range(len(self.y_train))))
        elif partition == "test":
            index = random.choice(list(range(len(self.y_test))))
            while self.y_train[index] != emotion:
                index = random.choice(list(range(len(self.y_test))))
        else:
            raise TypeError(
                "Unknown partition, only 'train' or 'test' is accepted")

        return index


def plot_histograms(classifiers=True, beta=0.5, n_classes=3, verbose=1):
    estimators = get_best_estimators(classifiers)

    final_result = {}
    for estimator, params, cv_score in estimators:
        final_result[estimator.__class__.__name__] = []
        for i in range(3):
            result = {}
            detector = EmotionRecognizer(estimator, verbose=0)
            detector.load_data()
            if i == 0:
                sample_size = 0.01
            elif i == 1:

                sample_size = 0.1
            elif i == 2:
                sample_size = 1
            n_train_samples = int(len(detector.X_train) * sample_size)
            n_test_samples = int(len(detector.X_test) * sample_size)
            detector.X_train = detector.X_train[:n_train_samples]
            detector.X_test = detector.X_test[:n_test_samples]
            detector.y_train = detector.y_train[:n_train_samples]
            detector.y_test = detector.y_test[:n_test_samples]
            t_train = time()
            detector.train()
            t_train = time() - t_train
            t_test = time()
            test_accuracy = detector.test_score()
            t_test = time() - t_test
            result['train_time'] = t_train
            result['pred_time'] = t_test
            result['acc_train'] = cv_score
            result['acc_test'] = test_accuracy
            result['f_train'] = detector.train_fbeta_score(beta)
            result['f_test'] = detector.test_fbeta_score(beta)
            if verbose:
                print(
                    f"[+] {estimator.__class__.__name__} with {sample_size*100}% ({n_train_samples}) data samples achieved {cv_score*100:.3f}% Validation Score in {t_train:.3f}s & {test_accuracy*100:.3f}% Test Score in {t_test:.3f}s"
                )
            final_result[estimator.__class__.__name__].append(result)
        if verbose:
            print()
    visualize(final_result, n_classes=n_classes)


def visualize(results, n_classes):

    n_estimators = len(results)

    accuracy = 1 / n_classes
    f1 = 1 / n_classes
    fig, ax = pl.subplots(2, 4, figsize=(11, 7))
    bar_width = 0.4
    colors = [(random.random(), random.random(), random.random())
              for _ in range(n_estimators)]
    for k, learner in enumerate(results.keys()):
        for j, metric in enumerate([
                'train_time', 'acc_train', 'f_train', 'pred_time', 'acc_test',
                'f_test'
        ]):
            for i in np.arange(3):
                x = bar_width * n_estimators
                ax[j // 3, j % 3].bar(i * x + k * (bar_width),
                                      results[learner][i][metric],
                                      width=bar_width,
                                      color=colors[k])
                ax[j // 3,
                   j % 3].set_xticks([x - 0.2, x * 2 - 0.2, x * 3 - 0.2])
                ax[j // 3, j % 3].set_xticklabels(["1%", "10%", "100%"])
                ax[j // 3, j % 3].set_xlabel("Training Set Size")
                ax[j // 3, j % 3].set_xlim((-0.2, x * 3))
    ax[0, 0].set_ylabel("Time (in seconds)")
    ax[0, 1].set_ylabel("Accuracy Score")
    ax[0, 2].set_ylabel("F-score")
    ax[1, 0].set_ylabel("Time (in seconds)")
    ax[1, 1].set_ylabel("Accuracy Score")
    ax[1, 2].set_ylabel("F-score")
    ax[0, 0].set_title("Model Training")
    ax[0, 1].set_title("Accuracy Score on Training Subset")
    ax[0, 2].set_title("F-score on Training Subset")
    ax[1, 0].set_title("Model Predicting")
    ax[1, 1].set_title("Accuracy Score on Testing Set")
    ax[1, 2].set_title("F-score on Testing Set")
    ax[0, 1].axhline(y=accuracy,
                     xmin=-0.1,
                     xmax=3.0,
                     linewidth=1,
                     color='k',
                     linestyle='dashed')
    ax[1, 1].axhline(y=accuracy,
                     xmin=-0.1,
                     xmax=3.0,
                     linewidth=1,
                     color='k',
                     linestyle='dashed')
    ax[0, 2].axhline(y=f1,
                     xmin=-0.1,
                     xmax=3.0,
                     linewidth=1,
                     color='k',
                     linestyle='dashed')
    ax[1, 2].axhline(y=f1,
                     xmin=-0.1,
                     xmax=3.0,
                     linewidth=1,
                     color='k',
                     linestyle='dashed')
    ax[0, 1].set_ylim((0, 1))
    ax[0, 2].set_ylim((0, 1))
    ax[1, 1].set_ylim((0, 1))
    ax[1, 2].set_ylim((0, 1))
    ax[0, 3].set_visible(False)
    ax[1, 3].axis('off')
    for i, learner in enumerate(results.keys()):
        pl.bar(0, 0, color=colors[i], label=learner)
    pl.legend()
    pl.suptitle("Performance Metrics for Three Supervised Learning Models",
                fontsize=16,
                y=1.10)
    pl.tight_layout()
    pl.show()
