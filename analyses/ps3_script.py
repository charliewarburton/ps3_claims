# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dask_ml.preprocessing import Categorizer
from glum import GeneralizedLinearRegressor, TweedieDistribution
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.metrics import auc
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from ps3.data import create_sample_split, load_transform

# %%
import ps3.data._sample_split as sample_split
from importlib import reload

reload(sample_split)

# %%
# load data
df = load_transform()

# %%
# Train benchmark tweedie model. This is entirely based on the glum tutorial.
weight = df["Exposure"].values
df["PurePremium"] = df["ClaimAmountCut"] / df["Exposure"]
y = df["PurePremium"]
# TODO: Why do you think, we divide by exposure here to arrive at our outcome variable?
# Exposure - How long (in years) policy held
# ClaimAmountCut - Total (cut) claim amount per policy

# PurePremium - Cost to insurer per year of the policy.

# So to get Pure Premium need to divide claim amount by exposure

# TODO: use your create_sample_split function here
df = sample_split.create_sample_split(df, "IDpol", 0.8)
train = np.where(df["sample"] == "train")
test = np.where(df["sample"] == "test")
df_train = df.iloc[train].copy()
df_test = df.iloc[test].copy()

categoricals = ["VehBrand", "VehGas", "Region", "Area", "DrivAge", "VehAge", "VehPower"]

predictors = categoricals + ["BonusMalus", "Density"]
glm_categorizer = Categorizer(columns=categoricals)

X_train_t = glm_categorizer.fit_transform(df[predictors].iloc[train])
X_test_t = glm_categorizer.transform(df[predictors].iloc[test])
y_train_t, y_test_t = y.iloc[train], y.iloc[test]
w_train_t, w_test_t = weight[train], weight[test]

TweedieDist = TweedieDistribution(1.5)
t_glm1 = GeneralizedLinearRegressor(family=TweedieDist, l1_ratio=1, fit_intercept=True)
t_glm1.fit(X_train_t, y_train_t, sample_weight=w_train_t)


pd.DataFrame(
    {"coefficient": np.concatenate(([t_glm1.intercept_], t_glm1.coef_))},
    index=["intercept"] + t_glm1.feature_names_,
).T

df_test["pp_t_glm1"] = t_glm1.predict(X_test_t)
df_train["pp_t_glm1"] = t_glm1.predict(X_train_t)

print(
    "training loss t_glm1:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_glm1"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_glm1:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_glm1"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

print(
    "Total claim amount on test set, observed = {}, predicted = {}".format(
        df["ClaimAmountCut"].values[test].sum(),
        np.sum(df["Exposure"].values[test] * t_glm1.predict(X_test_t)),
    )
)
# %%
# TODO: Let's add splines for BonusMalus and Density and use a Pipeline.
# Steps:
# 1. Define a Pipeline which chains a StandardScaler and SplineTransformer.
#    Choose knots="quantile" for the SplineTransformer and make sure, we
#    are only including one intercept in the final GLM.
# 2. Put the transforms together into a ColumnTransformer. Here we use OneHotEncoder for the categoricals.
# 3. Chain the transforms together with the GLM in a Pipeline.

# Let's put together a pipeline
numeric_cols = [
    "BonusMalus",
    "Density",
]  # Define numeric cols so can apply transformations
preprocessor = ColumnTransformer(  # Column transformer allows for different transformations for different columns
    transformers=[
        (
            "num",
            Pipeline(
                steps=[
                    ("scaler", StandardScaler()),  # Converts to Z scores
                    (
                        "spline",
                        SplineTransformer(
                            knots="quantile", n_knots=4, degree=3, include_bias=False
                        ),
                    ),  # Splines allow for non-linear relationships
                ]
            ),
            numeric_cols,
        ),
        (
            "cat",
            OneHotEncoder(sparse_output=False, drop="first"),
            categoricals,
        ),  # One hot encoding for categorical variables
    ]
)
preprocessor.set_output(transform="pandas")
# The pipeline is a sequence of steps.
# First step is preprocessor based on column transformer
# Second step is the estimator
model_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        # Fit the GLM with the preprocessed data
        (
            "estimate",
            GeneralizedLinearRegressor(
                family=TweedieDist,
                l1_ratio=1,  # l1 ratio = 1 gives lasso like regularization
                fit_intercept=True,
            ),
        ),
    ]
    # TODO: Define pipeline steps here
)

# let's have a look at the pipeline
model_pipeline

# let's check that the transforms worked
model_pipeline[:-1].fit_transform(df_train)

model_pipeline.fit(df_train, y_train_t, estimate__sample_weight=w_train_t)

pd.DataFrame(
    {
        "coefficient": np.concatenate(
            ([model_pipeline[-1].intercept_], model_pipeline[-1].coef_)
        )
    },
    index=["intercept"] + model_pipeline[-1].feature_names_,
).T

df_test["pp_t_glm2"] = model_pipeline.predict(df_test)
df_train["pp_t_glm2"] = model_pipeline.predict(df_train)

print(
    "training loss t_glm2:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_glm2"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_glm2:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_glm2"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

print(
    "Total claim amount on test set, observed = {}, predicted = {}".format(
        df["ClaimAmountCut"].values[test].sum(),
        np.sum(df["Exposure"].values[test] * df_test["pp_t_glm2"]),
    )
)

# %%
# TODO: Let's use a GBM instead as an estimator.
# Steps
# 1: Define the modelling pipeline. Tip: This can simply be a LGBMRegressor based on X_train_t from before.
# 2. Make sure we are choosing the correct objective for our estimator.

lgbm_estimate = LGBMRegressor(objective="tweedie")

# LGBM handles raw data well so no need for preprocessor
model_pipeline = Pipeline(steps=[("estimate", lgbm_estimate)])

model_pipeline.fit(X_train_t, y_train_t, estimate__sample_weight=w_train_t)
df_test["pp_t_lgbm"] = model_pipeline.predict(X_test_t)
df_train["pp_t_lgbm"] = model_pipeline.predict(X_train_t)
print(
    "training loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_lgbm"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_lgbm"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

# %%
# TODO: Let's tune the LGBM to reduce overfitting.
# Steps:
# 1. Define a `GridSearchCV` object with our lgbm pipeline/estimator. Tip: Parameters for a specific step of the pipeline
# can be passed by <step_name>__param.

# Note: Typically we tune many more parameters and larger grids,
# but to save compute time here, we focus on getting the learning rate
# and the number of estimators somewhat aligned -> tune learning_rate and n_estimators
cv = GridSearchCV(
    estimator=LGBMRegressor(objective="tweedie", tweedie_variance_power=1.5),
    param_grid={"learning_rate": [0.01, 0.1, 0.3], "n_estimators": [50, 100, 200]},
    cv=5,
    scoring="neg_mean_squared_error",
)
cv.fit(X_train_t, y_train_t, sample_weight=w_train_t)

df_test["pp_t_lgbm"] = cv.best_estimator_.predict(X_test_t)
df_train["pp_t_lgbm"] = cv.best_estimator_.predict(X_train_t)

print(
    "training loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_lgbm"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_lgbm"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

print(
    "Total claim amount on test set, observed = {}, predicted = {}".format(
        df["ClaimAmountCut"].values[test].sum(),
        np.sum(df["Exposure"].values[test] * df_test["pp_t_lgbm"]),
    )
)
# %%
# Let's compare the sorting of the pure premium predictions
# Lorenz curve shows how the models assign expected claim amounts to policyholders
# Closer to 'Oracle' - true amount - the better the model


# Source: https://scikit-learn.org/stable/auto_examples/linear_model/plot_tweedie_regression_insurance_claims.html
def lorenz_curve(y_true, y_pred, exposure):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    exposure = np.asarray(exposure)

    # order samples by increasing predicted risk:
    ranking = np.argsort(y_pred)
    ranked_exposure = exposure[ranking]
    ranked_pure_premium = y_true[ranking]
    cumulated_claim_amount = np.cumsum(ranked_pure_premium * ranked_exposure)
    cumulated_claim_amount /= cumulated_claim_amount[-1]
    cumulated_samples = np.linspace(0, 1, len(cumulated_claim_amount))
    return cumulated_samples, cumulated_claim_amount


fig, ax = plt.subplots(figsize=(8, 8))

for label, y_pred in [
    ("LGBM", df_test["pp_t_lgbm"]),
    ("GLM Benchmark", df_test["pp_t_glm1"]),
    ("GLM Splines", df_test["pp_t_glm2"]),
]:
    ordered_samples, cum_claims = lorenz_curve(
        df_test["PurePremium"], y_pred, df_test["Exposure"]
    )
    gini = 1 - 2 * auc(ordered_samples, cum_claims)
    label += f" (Gini index: {gini: .3f})"
    ax.plot(ordered_samples, cum_claims, linestyle="-", label=label)

# Oracle model: y_pred == y_test
ordered_samples, cum_claims = lorenz_curve(
    df_test["PurePremium"], df_test["PurePremium"], df_test["Exposure"]
)
gini = 1 - 2 * auc(ordered_samples, cum_claims)
label = f"Oracle (Gini index: {gini: .3f})"
ax.plot(ordered_samples, cum_claims, linestyle="-.", color="gray", label=label)

# Random baseline
ax.plot([0, 1], [0, 1], linestyle="--", color="black", label="Random baseline")
ax.set(
    title="Lorenz Curves",
    xlabel="Fraction of policyholders\n(ordered by model from safest to riskiest)",
    ylabel="Fraction of total claim amount",
)
ax.legend(loc="upper left")
plt.plot()

# %%
# Bin BonusMalus into quantiles (like we did with splines)
df_test["Quantile"] = pd.cut(
    df_test["BonusMalus"], bins=4, labels=["B1", "B2", "B3", "B4"]
)

# Compute weighted average claims for each quantile
def weighted_avg(group):
    return np.sum(group["ClaimAmountCut"] * group["Exposure"]) / np.sum(
        group["Exposure"]
    )


# Apply the weighted average calculation for each quantile
weighted_average_claims = (
    df_test.groupby("Quantile").apply(weighted_avg).reset_index(name="WeightedClaims")
)

# Plot the weighted average claims per quantile
plt.figure(figsize=(8, 5))
sns.barplot(
    x="Quantile", y="WeightedClaims", data=weighted_average_claims, palette="viridis"
)
plt.title("Weighted Average Claims Across Quantiles of BonusMalus", fontsize=14)
plt.xlabel("BonusMalus Quantiles", fontsize=12)
plt.ylabel("Weighted Average Claims", fontsize=12)
plt.show()
# Output: See that as BonusMalus increases, average claim increases. If no monotonicty constraint, model could predict decreasing claims as BonusMalus increases

# Check how many features we have in the preprocessor
preprocessor.fit(X_train_t)
feature_names = preprocessor.get_feature_names_out()
print(len(feature_names))  # output: 60

monotone_constraints = [1] + [0] * (len(feature_names) - 1)
constrained_lgbm = LGBMRegressor(
    objective="tweedie",
    tweedie_variance_power=1.5,
    mc=monotone_constraints,
    monotone_constraints_method="basic",
)
model_pipeline = Pipeline(
    steps=[("preprocessor", preprocessor), ("estimate", constrained_lgbm)]
)

model_pipeline.fit(X_train_t, y_train_t, estimate__sample_weight=w_train_t)

param_grid = {
    "estimate__learning_rate": [0.01, 0.1],
    "estimate__n_estimators": [100, 200],
}

cv = GridSearchCV(model_pipeline, param_grid, cv=5, scoring="neg_mean_poisson_deviance")
cv.fit(X_train_t, y_train_t, estimate__sample_weight=w_train_t)

df_test["pp_t_lgbm_constrained"] = cv.best_estimator_.predict(X_test_t)
df_train["pp_t_lgbm_constrained"] = cv.best_estimator_.predict(X_train_t)

print(
    "training loss t_lgbm_constrained:  {}".format(
        TweedieDist.deviance(
            y_train_t, df_train["pp_t_lgbm_constrained"], sample_weight=w_train_t
        )
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_lgbm_constrained:  {}".format(
        TweedieDist.deviance(
            y_test_t, df_test["pp_t_lgbm_constrained"], sample_weight=w_test_t
        )
        / np.sum(w_test_t)
    )
)

print(
    "Total claim amount on test set, observed = {}, predicted = {}".format(
        df["ClaimAmountCut"].values[test].sum(),
        np.sum(df["Exposure"].values[test] * df_test["pp_t_lgbm_constrained"]),
    )
)

#%%
# Plot learning curve
# Refit the best estimator with eval_set
best_lgbm = cv.best_estimator_.named_steps["estimate"]

# The code is crashing at this point
# Reason: [LightGBM] [Fatal] The output cannot be monotone with respect to categorical features
# best_lgbm.fit(
#     X_train_t,
#     y_train_t,
#     sample_weight=w_train_t,
#     eval_set=[(X_train_t, y_train_t), (X_test_t, y_test_t)],
#     eval_sample_weight=[w_train_t, w_test_t],
#     eval_metric="tweedie_deviance"
# )

# # Plot the learning curve
# import lightgbm as lgb
# lgb.plot_metric(best_lgbm, metric="tweedie_deviance")
# plt.title("Learning Curve for Constrained LGBM Regressor")
# plt.show()


# %%
