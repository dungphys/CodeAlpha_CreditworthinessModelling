from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import shap
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_PATH = ROOT / "data" / "processed-data" / "processed_data.csv"
RANDOM_STATE = 42


def load_train_test(
    data_path: Path = PROCESSED_DATA_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    df = pd.read_csv(data_path)
    y = df.pop("loan_status")
    return train_test_split(
        df,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )


class BaselineModel:
    def __init__(self, data_path: Path = PROCESSED_DATA_PATH):
        self.data_path = Path(data_path)
        self.X_train, self.X_test, self.y_train, self.y_test = load_train_test(
            self.data_path
        )

        binary_features = [
            "high_lti",
            "loan_grade_sep",
            "high_income",
            "high_lti_x_rent",
            "cb_person_default_on_file",
        ]
        self.numeric_features = self.X_train.select_dtypes(
            include=np.number
        ).columns.tolist()
        self.numeric_features = [
            feature for feature in self.numeric_features
            if feature not in binary_features
        ]
        self.ohe_features = self.X_train.select_dtypes(
            exclude=np.number
        ).columns.tolist()

        self.impute_preprocessor = ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    Pipeline(
                        [
                            ("scaler", StandardScaler()),
                            (
                                "imputer",
                                KNNImputer(
                                    n_neighbors=5,
                                    weights="distance",
                                    add_indicator=True,
                                ),
                            ),
                        ]
                    ),
                    self.numeric_features,
                ),
                (
                    "ohe",
                    Pipeline(
                        [
                            (
                                "encoder",
                                OneHotEncoder(
                                    sparse_output=False, handle_unknown="ignore"
                                ),
                            )
                        ]
                    ),
                    self.ohe_features,
                ),
            ],
            remainder="passthrough",
        )
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", "passthrough", self.numeric_features),
                (
                    "ohe",
                    Pipeline(
                        [
                            (
                                "encoder",
                                OneHotEncoder(
                                    sparse_output=False, handle_unknown="ignore"
                                ),
                            )
                        ]
                    ),
                    self.ohe_features,
                ),
            ],
            remainder="passthrough",
        )

    def _report(self, model: Pipeline) -> tuple[dict, list[str]]:
        positive_idx = list(model.classes_).index(1)
        rows = {}
        reports = []
        for set_name, features, target in [
            ("Training", self.X_train, self.y_train),
            ("Evaluation", self.X_test, self.y_test),
        ]:
            prediction = model.predict(features)
            score = model.predict_proba(features)[:, positive_idx]
            rows[set_name] = {
                "prevalence": np.mean(target),
                "PR-AUC (AP)": average_precision_score(target, score),
                "ROC-AUC": roc_auc_score(target, score),
                "balanced acc": balanced_accuracy_score(target, prediction),
                "F1 (pos)": f1_score(target, prediction, zero_division=0),
            }
            reports.append(
                classification_report(
                    target, prediction, digits=3, zero_division=0
                )
            )
            if set_name == "Evaluation":
                print(f"Test PR-AUC: {average_precision_score(target, score):.3f}")
                print("Confusion matrix (rows = true, cols = predicted):")
                ConfusionMatrixDisplay.from_predictions(
                    target, prediction, cmap=plt.cm.Blues
                )
                plt.show()
        return rows, reports

    def _bootstrap_ci(self, model: Pipeline, n_bootstrap: int = 2000) -> pd.DataFrame:
        y_true = self.y_test.to_numpy()
        y_pred = model.predict(self.X_test)
        positive_idx = list(model.classes_).index(1)
        y_score = model.predict_proba(self.X_test)[:, positive_idx]

        def calculate_metrics(y, pred, score):
            return {
                "PR-AUC (AP)": average_precision_score(y, score),
                "ROC-AUC": roc_auc_score(y, score),
                "Balanced accuracy": balanced_accuracy_score(y, pred),
                "Accuracy": accuracy_score(y, pred),
                "Precision": precision_score(y, pred, pos_label=1, zero_division=0),
                "Precision (macro)": precision_score(
                    y, pred, average="macro", zero_division=0
                ),
                "Recall": recall_score(y, pred, pos_label=1, zero_division=0),
                "F1": f1_score(y, pred, pos_label=1, zero_division=0),
                "F1 (macro)": f1_score(y, pred, average="macro", zero_division=0),
            }

        point_estimates = calculate_metrics(y_true, y_pred, y_score)
        rng = np.random.default_rng(RANDOM_STATE)
        class_indices = [
            np.flatnonzero(y_true == label) for label in np.unique(y_true)
        ]
        boot_scores = {name: [] for name in point_estimates}

        for _ in range(n_bootstrap):
            sampled_idx = np.concatenate(
                [
                    rng.choice(indices, size=len(indices), replace=True)
                    for indices in class_indices
                ]
            )
            scores = calculate_metrics(
                y_true[sampled_idx],
                y_pred[sampled_idx],
                y_score[sampled_idx],
            )
            for name, value in scores.items():
                boot_scores[name].append(value)

        return pd.DataFrame(
            {
                name: {
                    "estimate": estimate,
                    "95% CI lower": np.percentile(boot_scores[name], 2.5),
                    "95% CI upper": np.percentile(boot_scores[name], 97.5),
                }
                for name, estimate in point_estimates.items()
            }
        ).T.round(3)

    def run(self) -> dict[str, Pipeline]:
        dummy_models = {
            "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
            "dummy_stratified": DummyClassifier(
                strategy="stratified", random_state=RANDOM_STATE
            ),
            "dummy_constant": DummyClassifier(strategy="constant", constant=1),
        }
        for name, model in dummy_models.items():
            model.fit(self.X_train, self.y_train)
            prediction = model.predict(self.X_test)
            positive_idx = list(model.classes_).index(1)
            score = model.predict_proba(self.X_test)[:, positive_idx]
            print(f"\n{name}")
            print(f"Average precision: {average_precision_score(self.y_test, score):.4f}")
            print(
                "Balanced accuracy: "
                f"{balanced_accuracy_score(self.y_test, prediction):.4f}"
            )
            print(f"ROC AUC: {roc_auc_score(self.y_test, score):.4f}")
            print(f"F1 score: {f1_score(self.y_test, prediction, zero_division=0):.4f}")
            print(
                classification_report(
                    self.y_test, prediction, digits=3, zero_division=0
                )
            )

        tree_params = {
            "max_depth": 5,
            "min_samples_leaf": 20,
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
        }
        impute_model = Pipeline(
            [
                ("preprocessor", self.impute_preprocessor),
                ("classifier", DecisionTreeClassifier(**tree_params)),
            ]
        ).fit(self.X_train, self.y_train)
        model = Pipeline(
            [
                ("preprocessor", self.preprocessor),
                ("classifier", DecisionTreeClassifier(**tree_params)),
            ]
        ).fit(self.X_train, self.y_train)

        for name, fitted_model in [
            ("With kNN imputation", impute_model),
            ("Without kNN imputation", model),
        ]:
            rows, reports = self._report(fitted_model)
            print(f"\n{name} metrics:")
            print(pd.DataFrame(rows).T.round(4).to_string())
            for set_name, report in zip(["train", "test"], reports):
                print(f"Set {set_name}\n{report}")
            print(f"\n{name} 95% confidence intervals:")
            print(self._bootstrap_ci(fitted_model))

        pre = model.named_steps["preprocessor"]
        tree = model.named_steps["classifier"]
        names = pre.get_feature_names_out()
        print(export_text(tree, feature_names=list(names), show_weights=True))
        fig, ax = plt.subplots(figsize=(22, 10))
        fig.suptitle("Decision tree without kNN imputation")
        plot_tree(
            tree,
            feature_names=names,
            class_names=["neg", "pos"],
            filled=True,
            impurity=False,
            proportion=True,
            fontsize=8,
            ax=ax,
        )
        plt.show()
        print(
            pd.Series(tree.feature_importances_, index=names)
            .sort_values(ascending=False)
            .head(15)
        )
        pd.Series(tree.feature_importances_, index=names).nlargest(12).sort_values().plot(
            kind="barh", title="Feature importance: without-kNN case"
        )
        plt.tight_layout()
        plt.show()

        impute_pre = impute_model.named_steps["preprocessor"]
        impute_tree = impute_model.named_steps["classifier"]
        impute_names = impute_pre.get_feature_names_out()
        pd.Series(
            impute_tree.feature_importances_, index=impute_names
        ).nlargest(12).sort_values().plot(
            kind="barh", title="Feature importance: with-kNN case"
        )
        plt.tight_layout()
        plt.show()
        return {"with_knn": impute_model, "without_knn": model}


class LogisticRegressionModel:
    def __init__(
        self,
        data_path: Path = PROCESSED_DATA_PATH,
        n_trials: int = 120,
    ):
        self.data_path = Path(data_path)
        self.n_trials = n_trials
        self.X_train, self.X_test, self.y_train, self.y_test = load_train_test(
            self.data_path
        )
        binary_features = [
            "high_lti",
            "loan_grade_sep",
            "high_income",
            "high_lti_x_rent",
            "cb_person_default_on_file",
        ]
        self.numeric_features = self.X_train.select_dtypes(
            include=np.number
        ).columns.tolist()
        self.numeric_features = [
            feature for feature in self.numeric_features
            if feature not in binary_features
        ]
        self.ohe_features = self.X_train.select_dtypes(
            exclude=np.number
        ).columns.tolist()
        self.cv = StratifiedKFold(
            n_splits=5, shuffle=True, random_state=RANDOM_STATE
        )

    def _make_pipeline(
        self,
        scaler,
        n_neighbors: int,
        weights: str,
        C: float,
        solver: str,
        l1_ratio: float,
        pos_weight: float,
    ) -> Pipeline:
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    Pipeline(
                        [
                            ("scaler", scaler),
                            (
                                "imputer",
                                KNNImputer(
                                    n_neighbors=n_neighbors,
                                    weights=weights,
                                    add_indicator=True,
                                ),
                            ),
                        ]
                    ),
                    self.numeric_features,
                ),
                (
                    "ohe",
                    Pipeline(
                        [("encoder", OneHotEncoder(handle_unknown="ignore"))]
                    ),
                    self.ohe_features,
                ),
            ],
            remainder="passthrough",
        )
        return Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    LogisticRegression(
                        C=C,
                        solver=solver,
                        l1_ratio=l1_ratio,
                        class_weight={0: 1.0, 1: pos_weight},
                        max_iter=2000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )

    def optuna_study(self, metric: str = "f1") -> optuna.Study:
        def objective(trial):
            scaler_ind = trial.suggest_categorical(
                "scaler_ind", ["standard", "robust"]
            )
            scaler = StandardScaler() if scaler_ind == "standard" else RobustScaler()
            knn_n_neighbors = trial.suggest_int("knn_n_neighbors", 2, 40)
            knn_weights = trial.suggest_categorical(
                "knn_weights", ["uniform", "distance"]
            )
            imbalance_ratio = (self.y_train == 0).sum() / (self.y_train == 1).sum()
            pos_weight = trial.suggest_float("pos_weight", 1.0, 2 * imbalance_ratio)
            logit_C = trial.suggest_float("logit_C", 1e-4, 1e2, log=True)
            solver = trial.suggest_categorical(
                "logit_solver",
                ["lbfgs", "newton-cholesky", "liblinear", "saga"],
            )
            if solver in ["lbfgs", "newton-cholesky"]:
                l1_ratio = 0.0
            elif solver == "liblinear":
                l1_ratio = trial.suggest_categorical(
                    "logit_l1_ratio", [0.0, 1.0]
                )
            else:
                l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)

            pipeline = self._make_pipeline(
                scaler,
                knn_n_neighbors,
                knn_weights,
                logit_C,
                solver,
                l1_ratio,
                pos_weight,
            )
            if metric == "f1":
                threshold = trial.suggest_float("threshold", 0.15, 0.75)

                def f1_at_threshold(estimator, features, target):
                    probability = estimator.predict_proba(features)[:, 1]
                    return f1_score(
                        target, (probability >= threshold).astype(int)
                    )

                scores = cross_val_score(
                    pipeline,
                    self.X_train,
                    self.y_train,
                    scoring=f1_at_threshold,
                    cv=self.cv,
                    n_jobs=-1,
                )
            else:
                scores = cross_val_score(
                    pipeline,
                    self.X_train,
                    self.y_train,
                    scoring=metric,
                    cv=self.cv,
                    n_jobs=-1,
                )
            return scores.mean()

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        )
        study.optimize(objective, n_trials=self.n_trials)
        return study

    def retrain_and_evaluate(
        self, study: optuna.Study, metric: str = "f1"
    ) -> Pipeline:
        params = study.best_params
        scaler_map = {
            "standard": StandardScaler,
            "robust": RobustScaler,
        }
        pipeline = self._make_pipeline(
            scaler_map[params["scaler_ind"]](),
            params["knn_n_neighbors"],
            params["knn_weights"],
            params["logit_C"],
            params["logit_solver"],
            params.get("logit_l1_ratio", params.get("l1_ratio", 0.0)),
            params["pos_weight"],
        )
        pipeline.fit(self.X_train, self.y_train)
        threshold = params.get("threshold", 0.5)

        print(f"Best CV {metric.upper()}: {study.best_value:.4f}")
        for set_name, features, target in [
            ("Test", self.X_test, self.y_test),
            ("Training", self.X_train, self.y_train),
        ]:
            probability = pipeline.predict_proba(features)[:, 1]
            prediction = (probability >= threshold).astype(int)
            print(f"\n{set_name} F1:                {f1_score(target, prediction):.4f}")
            print(
                f"{set_name} balanced accuracy: "
                f"{balanced_accuracy_score(target, prediction):.4f}"
            )
            print(
                f"{set_name} ROC-AUC:           "
                f"{roc_auc_score(target, probability):.4f}"
            )
            print(
                f"{set_name} average precision: "
                f"{average_precision_score(target, probability):.4f}"
            )
            print(f"\n{set_name} classification report:")
            print(
                classification_report(
                    target, prediction, zero_division=0, digits=3
                )
            )
            ConfusionMatrixDisplay.from_predictions(target, prediction)
            plt.show()
        return pipeline

    def bootstrap_ci(
        self,
        pipeline: Pipeline,
        threshold: float,
        n_bootstrap: int = 2000,
    ) -> pd.DataFrame:
        y_true = self.y_test.to_numpy()
        y_prob = pipeline.predict_proba(self.X_test)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)
        class_indices = [
            np.flatnonzero(y_true == label) for label in np.unique(y_true)
        ]
        rng = np.random.default_rng(RANDOM_STATE)

        def calculate_metrics(indices):
            target = y_true[indices]
            prediction = y_pred[indices]
            probability = y_prob[indices]
            return {
                "F1": f1_score(target, prediction, zero_division=0),
                "Precision (macro)": precision_score(
                    target, prediction, average="macro", zero_division=0
                ),
                "F1 (macro)": f1_score(
                    target, prediction, average="macro", zero_division=0
                ),
                "Precision": precision_score(target, prediction, zero_division=0),
                "Recall": recall_score(target, prediction, zero_division=0),
                "Accuracy": accuracy_score(target, prediction),
                "Balanced accuracy": balanced_accuracy_score(target, prediction),
                "ROC-AUC": roc_auc_score(target, probability),
                "Average precision": average_precision_score(target, probability),
            }

        point_estimates = calculate_metrics(np.arange(len(y_true)))
        bootstrap_results = {name: [] for name in point_estimates}
        for _ in range(n_bootstrap):
            sampled_indices = np.concatenate(
                [
                    rng.choice(indices, size=len(indices), replace=True)
                    for indices in class_indices
                ]
            )
            for name, value in calculate_metrics(sampled_indices).items():
                bootstrap_results[name].append(value)

        return pd.DataFrame(
            {
                "Estimate": point_estimates,
                "95% CI lower": {
                    name: np.percentile(values, 2.5)
                    for name, values in bootstrap_results.items()
                },
                "95% CI upper": {
                    name: np.percentile(values, 97.5)
                    for name, values in bootstrap_results.items()
                },
            }
        ).round(4)

    def explain(self, pipeline: Pipeline) -> None:
        background_raw = self.X_train
        explain_raw = self.X_test
        preprocessor = pipeline.named_steps["preprocessor"]
        classifier = pipeline.named_steps["classifier"]
        feature_names = preprocessor.get_feature_names_out()
        background = preprocessor.transform(background_raw)
        explain_data = preprocessor.transform(explain_raw)
        if hasattr(background, "toarray"):
            background = background.toarray()
        if hasattr(explain_data, "toarray"):
            explain_data = explain_data.toarray()
        background = pd.DataFrame(
            background, columns=feature_names, index=background_raw.index
        )
        explain_data = pd.DataFrame(
            explain_data, columns=feature_names, index=explain_raw.index
        )

        masker = shap.maskers.Independent(
            background, max_samples=len(background)
        )
        explainer = shap.LinearExplainer(classifier, masker)
        shap_values = explainer(explain_data)

        groups = np.array(
            [
                self._to_original(feature, self.X_train.columns)
                for feature in feature_names
            ]
        )
        aggregated = (
            pd.DataFrame(shap_values.values, columns=feature_names)
            .T.groupby(groups)
            .sum()
            .T
        )
        aggregated_explanation = shap.Explanation(
            values=aggregated.values,
            base_values=shap_values.base_values,
            feature_names=list(aggregated.columns),
        )
        shap.plots.bar(aggregated_explanation, max_display=15, show=False)
        plt.xlabel("Mean |SHAP value| (log-odds)")
        plt.tight_layout()
        plt.show()
        shap.plots.beeswarm(shap_values, max_display=20, show=False)
        plt.xlabel("SHAP value (impact on log-odds of the predicted class)")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def _to_original(feature_name: str, raw_columns) -> str:
        stripped = feature_name.split("__", 1)[-1]
        matches = [
            column
            for column in raw_columns
            if stripped == column or stripped.startswith(column + "_")
        ]
        return max(matches, key=len) if matches else stripped

    def run(self) -> dict:
        study = self.optuna_study(metric="f1")
        print("Best performing pipeline:")
        print(study.best_trial)
        print("\nBest params:")
        print(study.best_value, study.best_params)

        pipeline = self.retrain_and_evaluate(study, metric="f1")
        threshold = study.best_params.get("threshold", 0.5)
        print("\nTest-set bootstrap confidence intervals:")
        print(self.bootstrap_ci(pipeline, threshold))
        self.explain(pipeline)
        return {"study": study, "pipeline": pipeline, "threshold": threshold}


class XGBoostModel:
    def __init__(
        self,
        data_path: Path = PROCESSED_DATA_PATH,
        n_trials: int = 120,
    ):
        self.data_path = Path(data_path)
        self.n_trials = n_trials
        self.X_train, self.X_test, self.y_train, self.y_test = load_train_test(
            self.data_path
        )
        binary_features = [
            "high_lti",
            "loan_grade_sep",
            "high_income",
            "high_lti_x_rent",
            "cb_person_default_on_file",
        ]
        numeric_features = self.X_train.select_dtypes(
            include=np.number
        ).columns.tolist()
        self.numeric_features = [
            feature for feature in numeric_features
            if feature not in binary_features
        ]
        self.ohe_features = self.X_train.select_dtypes(
            exclude=np.number
        ).columns.tolist()
        self.cv = StratifiedKFold(
            n_splits=5, shuffle=True, random_state=RANDOM_STATE
        )
        self.preprocessor = ColumnTransformer(
            transformers=[
                (
                    "ohe",
                    Pipeline(
                        [("encoder", OneHotEncoder(handle_unknown="ignore"))]
                    ),
                    self.ohe_features,
                )
            ],
            remainder="passthrough",
        )
        self.folds = self._prepare_folds()
        self.beta = 1.0

    @staticmethod
    def best_fbeta(y_true, probability, beta: float = 1.0) -> tuple[float, float]:
        precision, recall, thresholds = precision_recall_curve(y_true, probability)
        beta_squared = beta**2
        f_score = (
            (1 + beta_squared) * precision * recall
            / (beta_squared * precision + recall + 1e-12)
        )
        best_index = np.argmax(f_score[:-1])
        return f_score[best_index], thresholds[best_index]

    def _prepare_folds(self) -> list:
        y_array = np.asarray(self.y_train)
        folds = []
        for train_indices, validation_indices in self.cv.split(
            self.X_train, y_array
        ):
            preprocessor = clone(self.preprocessor).fit(
                self.X_train.iloc[train_indices]
            )
            folds.append(
                (
                    preprocessor.transform(self.X_train.iloc[train_indices]),
                    y_array[train_indices],
                    preprocessor.transform(self.X_train.iloc[validation_indices]),
                    y_array[validation_indices],
                )
            )
        return folds

    def _params_set(self, trial: optuna.Trial) -> dict:
        return {
            "learning_rate": trial.suggest_float("eta", 5e-3, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 7),
            "min_child_weight": trial.suggest_float(
                "min_child_weight", 1.0, 20.0, log=True
            ),
            "subsample": trial.suggest_float("subsample", 0.4, 1.0),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.4, 1.0
            ),
            "gamma": trial.suggest_float("gamma", 2, 40, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 6, 20, log=True),
        }

    def objective(self, trial: optuna.Trial) -> float:
        params = self._params_set(trial)
        scores = []
        best_iterations = []
        for step, (X_train, y_train, X_valid, y_valid) in enumerate(self.folds):
            model = XGBClassifier(
                **params,
                n_estimators=850,
                early_stopping_rounds=50,
                eval_metric="aucpr",
                tree_method="hist",
                objective="binary:logistic",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )
            model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
            probability = model.predict_proba(X_valid)[:, 1]
            score, _ = self.best_fbeta(y_valid, probability, self.beta)
            scores.append(score)
            best_iterations.append(model.best_iteration + 1)
            trial.report(np.mean(scores), step)
            if trial.should_prune():
                raise optuna.TrialPruned()

        trial.set_user_attr("n_estimators", int(np.mean(best_iterations)))
        return float(np.mean(scores))

    def optuna_study(self) -> optuna.Study:
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(
                seed=RANDOM_STATE, multivariate=True
            ),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=15, n_warmup_steps=2
            ),
        )
        study.optimize(self.objective, n_trials=self.n_trials)
        return study

    def _oof_threshold(
        self, best_params: dict, n_estimators: int
    ) -> tuple[float, float]:
        oof_probabilities = []
        oof_targets = []
        for X_train, y_train, X_valid, y_valid in self.folds:
            model = XGBClassifier(
                **best_params,
                n_estimators=n_estimators,
                tree_method="hist",
                objective="binary:logistic",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ).fit(X_train, y_train)
            oof_probabilities.append(model.predict_proba(X_valid)[:, 1])
            oof_targets.append(y_valid)
        score, threshold = self.best_fbeta(
            np.concatenate(oof_targets),
            np.concatenate(oof_probabilities),
            self.beta,
        )
        return threshold, score

    def _report(self, name: str, pipeline: Pipeline, X, y, threshold: float) -> None:
        probability = pipeline.predict_proba(X)[:, 1]
        prediction = (probability >= threshold).astype(int)
        print(f"\n{name} F1:                {f1_score(y, prediction):.4f}")
        print(
            f"{name} balanced accuracy: "
            f"{balanced_accuracy_score(y, prediction):.4f}"
        )
        print(f"{name} ROC-AUC:           {roc_auc_score(y, probability):.4f}")
        print(
            f"{name} average precision: "
            f"{average_precision_score(y, probability):.4f}"
        )
        print(f"\n{name} classification report:")
        print(classification_report(y, prediction, zero_division=0, digits=3))
        ConfusionMatrixDisplay.from_predictions(y, prediction)
        plt.title(f"{name} (threshold = {threshold:.3f})")
        plt.show()

    def retrain_and_evaluate(
        self,
        study: optuna.Study,
        n_estimators_scale: float = 1.1,
    ) -> tuple[Pipeline, float]:
        best = study.best_trial
        params = dict(best.params)
        params["learning_rate"] = params.pop("eta")
        n_estimators = int(best.user_attrs["n_estimators"] * n_estimators_scale)
        threshold, oof_f1 = self._oof_threshold(
            params, best.user_attrs["n_estimators"]
        )
        pipeline = Pipeline(
            [
                ("preprocessor", clone(self.preprocessor)),
                (
                    "classifier",
                    XGBClassifier(
                        **params,
                        n_estimators=n_estimators,
                        tree_method="hist",
                        objective="binary:logistic",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
        pipeline.fit(self.X_train, self.y_train)
        print(f"Best CV f1: {study.best_value:.4f}")
        print(f"OOF F1 at tuned threshold: {oof_f1:.4f}")
        print(f"Threshold:                 {threshold:.3f}")
        print(f"n_estimators:              {n_estimators}")
        self._report("Test", pipeline, self.X_test, self.y_test, threshold)
        self._report("Training", pipeline, self.X_train, self.y_train, threshold)
        return pipeline, threshold

    def bootstrap_ci(
        self,
        pipeline: Pipeline,
        threshold: float,
        n_bootstrap: int = 2000,
    ) -> pd.DataFrame:
        y_true = self.y_test.to_numpy()
        y_prob = pipeline.predict_proba(self.X_test)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)
        point_estimates = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Balanced accuracy": balanced_accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1": f1_score(y_true, y_pred),
            "Precision (macro)": precision_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
            "F1 (macro)": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "ROC-AUC": roc_auc_score(y_true, y_prob),
            "Average precision": average_precision_score(y_true, y_prob),
        }
        rng = np.random.default_rng(RANDOM_STATE)
        metric_values = {name: np.empty(n_bootstrap) for name in point_estimates}
        classes, counts = np.unique(y_true, return_counts=True)
        class_indices = {
            label: np.flatnonzero(y_true == label) for label in classes
        }

        for i in range(n_bootstrap):
            sample_idx = np.concatenate(
                [
                    rng.choice(class_indices[label], size=count, replace=True)
                    for label, count in zip(classes, counts)
                ]
            )
            y_boot = y_true[sample_idx]
            prob_boot = y_prob[sample_idx]
            pred_boot = y_pred[sample_idx]
            metric_values["Accuracy"][i] = accuracy_score(y_boot, pred_boot)
            metric_values["Balanced accuracy"][i] = balanced_accuracy_score(
                y_boot, pred_boot
            )
            metric_values["Precision"][i] = precision_score(
                y_boot, pred_boot, zero_division=0
            )
            metric_values["Recall"][i] = recall_score(
                y_boot, pred_boot, zero_division=0
            )
            metric_values["F1"][i] = f1_score(y_boot, pred_boot)
            metric_values["Precision (macro)"][i] = precision_score(
                y_boot, pred_boot, average="macro", zero_division=0
            )
            metric_values["F1 (macro)"][i] = f1_score(
                y_boot, pred_boot, average="macro", zero_division=0
            )
            metric_values["ROC-AUC"][i] = roc_auc_score(y_boot, prob_boot)
            metric_values["Average precision"][i] = average_precision_score(
                y_boot, prob_boot
            )

        return pd.DataFrame(
            {
                "Metric": list(point_estimates),
                "Point estimate": list(point_estimates.values()),
                "95% CI lower": [
                    np.percentile(metric_values[name], 2.5)
                    for name in point_estimates
                ],
                "95% CI upper": [
                    np.percentile(metric_values[name], 97.5)
                    for name in point_estimates
                ],
            }
        )

    @staticmethod
    def _to_original(feature_name: str, raw_columns) -> str:
        stripped = feature_name.split("__", 1)[-1]
        matches = [
            column
            for column in raw_columns
            if stripped == column or stripped.startswith(column + "_")
        ]
        return max(matches, key=len) if matches else stripped

    def explain(self, pipeline: Pipeline) -> None:
        background_raw = self.X_train
        explain_raw = self.X_test
        rng = np.random.RandomState(RANDOM_STATE)
        background_indices = rng.choice(
            len(background_raw), size=min(400, len(background_raw)), replace=False
        )
        explain_indices = rng.choice(
            len(explain_raw), size=min(1200, len(explain_raw)), replace=False
        )
        preprocessor = pipeline.named_steps["preprocessor"]
        classifier = pipeline.named_steps["classifier"]
        feature_names = preprocessor.get_feature_names_out()
        background = preprocessor.transform(
            background_raw.iloc[background_indices]
        )
        explain_data = preprocessor.transform(explain_raw.iloc[explain_indices])
        if hasattr(background, "toarray"):
            background = background.toarray()
        if hasattr(explain_data, "toarray"):
            explain_data = explain_data.toarray()
        background = pd.DataFrame(
            background,
            columns=feature_names,
            index=background_raw.index[background_indices],
        )
        explain_data = pd.DataFrame(
            explain_data,
            columns=feature_names,
            index=explain_raw.index[explain_indices],
        )

        masker = shap.maskers.Independent(
            background, max_samples=len(background)
        )
        explainer = shap.TreeExplainer(classifier, masker)
        shap_values = explainer(explain_data)
        groups = np.array(
            [
                self._to_original(feature, self.X_train.columns)
                for feature in feature_names
            ]
        )
        aggregated = (
            pd.DataFrame(shap_values.values, columns=feature_names)
            .T.groupby(groups)
            .sum()
            .T
        )
        aggregated_explanation = shap.Explanation(
            values=aggregated.values,
            base_values=shap_values.base_values,
            feature_names=list(aggregated.columns),
        )
        shap.plots.bar(aggregated_explanation, max_display=15, show=False)
        plt.xlabel("Mean |SHAP value| (log-odds)")
        plt.tight_layout()
        plt.show()
        shap.plots.beeswarm(shap_values, max_display=20, show=False)
        plt.xlabel("SHAP value (impact on log-odds of the predicted class)")
        plt.tight_layout()
        plt.show()

    def run(self) -> dict:
        study = self.optuna_study()
        pipeline, threshold = self.retrain_and_evaluate(study)
        print("\nTest-set bootstrap confidence intervals:")
        print(self.bootstrap_ci(pipeline, threshold).round(4).to_string(index=False))
        print("\nParameter importance:")
        print(optuna.importance.get_param_importances(study))
        trials = study.trials_dataframe()
        top_trials = trials.sort_values("value", ascending=False).head(15)
        parameter_columns = [
            "value",
            "params_max_depth",
            "params_min_child_weight",
            "params_gamma",
            "params_subsample",
            "params_colsample_bytree",
            "params_reg_lambda",
        ]
        print(top_trials[parameter_columns].to_string(index=False))
        print(
            trials.groupby("params_max_depth")["value"]
            .agg(["max", "mean", "count"])
        )
        self.explain(pipeline)
        return {"study": study, "pipeline": pipeline, "threshold": threshold}


def main() -> None:
    BaselineModel().run()
    LogisticRegressionModel().run()
    XGBoostModel().run()


if __name__ == "__main__":
    main()
