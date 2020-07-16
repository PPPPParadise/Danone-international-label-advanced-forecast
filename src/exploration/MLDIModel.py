from sklearn.ensemble import ExtraTreesRegressor

class GlobalDIModel:

    def __init__(self, horizon, date_when_predicting, model_params):
        self.horizon = horizon
        self.date_when_predicting = date_when_predicting
        self.model_params = model_params

        #self.model = xgb.XGBRegressor(**self.model_params)
        self.model = ExtraTreesRegressor(**self.model_params)
        self.is_model_fitted = False

    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)
        self.is_model_fitted = True

    def predict(self, x):
        if not self.is_model_fitted:
            raise Exception('Model should be first fitted before predict method is called')
        return self.model.predict(x)

    def __repr__(self):
        return f"< Global DI Model | @horizon {self.horizon} | @dwp {self.date_when_predicting} >"
