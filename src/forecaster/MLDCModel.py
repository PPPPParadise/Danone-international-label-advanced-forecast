from copy import deepcopy
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, RANSACRegressor, TheilSenRegressor, HuberRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA


import xgboost as xgb


models_mapping = {
    "XGBRegressor": xgb.XGBRegressor,
    "RandomForestRegressor": RandomForestRegressor,
    "ExtraTreesRegressor": ExtraTreesRegressor,
    "AdaBoostRegressor": AdaBoostRegressor,
    "LinearRegression": LinearRegression,
    "Ridge": Ridge,
    "Lasso": Lasso,
    "ElasticNet": ElasticNet,
    "SVR": SVR,
    "GradientBoostingRegressor": GradientBoostingRegressor,
    "RANSACRegressor": RANSACRegressor,
    "TheilSenRegressor": TheilSenRegressor,
    "HuberRegressor": HuberRegressor,
    "PLSRegression": PLSRegression,
}


class MLDCModel:

    def __init__(self, model_name, model_params):

        self.model_name = model_name
        self.model_params = deepcopy(model_params)
        self.is_fitted = False

        # 1. Check model is supported
        if model_name not in models_mapping.keys():
            raise ValueError(f"model {model_name} is not yet supported. Only following"
                             f" models are supported : {' '.join(models_mapping.keys())}")

        # 2. Check that standrad scaling argument is specified
        if "standard_scaling" not in self.model_params.keys():
            self.model_params["standard_scaling"] = False

        # 3. Check that pca argument is specified
        if "pca" not in self.model_params.keys():
            self.model_params["pca"] = 0

        self.scale_features = self.model_params.pop("standard_scaling")
        self.scaler = StandardScaler()

        self.pca_param = self.model_params.pop("pca", 0)
        self.apply_pca = self.pca_param > 0
        self.pca = PCA(self.pca_param)

        self.model = models_mapping[model_name](**self.model_params)

    def fit(self, X, y):

        if self.scale_features:
            X = self.scaler.fit_transform(X)

        if self.apply_pca:
            X = self.pca.fit_transform(X)

        self.model.fit(X, y)
        self.is_fitted = True

    def predict(self, X):

        if not self.is_fitted:
            raise NotImplementedError(
                "Model is not trained - You should first train the model by calling the"
                " `fit` method before calling the `predict` method "
            )

        if self.scale_features:
            X = self.scaler.transform(X)

        if self.apply_pca:
            X = self.pca.transform(X)

        return self.model.predict(X)

    @property
    def feature_importances_(self):
        return self.model.feature_importances_
