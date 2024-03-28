import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error

def nonlinear_least_squares_fit(model, xdata, ydata, p0=None, bounds=(-np.inf, np.inf)):
    """
    Perform nonlinear least squares fitting using the specified model.

    Nonlinear least squares fitting is used to find the parameters of a nonlinear model
    that minimize the sum of the squares of the differences between the observed data and
    the corresponding model predictions.

    Args:
        model (callable): A callable function representing the model to be fitted.
        xdata (array-like): Independent variable data.
        ydata (array-like): Dependent variable data.
        p0 (array-like, optional): Initial guess for the parameters of the model.
        bounds (2-tuple of array-like, optional): Bounds on parameters for the curve fit.

    Returns:
        tuple: A tuple containing the optimal parameters of the model and the covariance matrix.
    """
    # Perform the curve fitting using scipy.optimize.curve_fit
    optimal_params, covariance = curve_fit(
        model, xdata, ydata, p0=p0, bounds=bounds)

    return optimal_params, covariance


def calculate_residuals(model, params, xdata, ydata):
    """
    Calculate residuals between the model predictions and the actual data.

    Residuals are the differences between the observed values and the values predicted by the model.

    Args:
        model (callable): The model function.
        params (array-like): The parameters of the model.
        x_data (array-like): The independent variable data.
        y_data (array-like): The observed dependent variable data.

    Returns:
        array-like: The residuals.
    """
    # Calculate the model predictions
    predictions = model(xdata, *params)

    # Calculate the residuals by subtracting the observed values from the predictions
    residuals = ydata - predictions

    return residuals


def calculate_mean_squared_error(model, params, xdata, ydata):
    """
    Calculate the mean squared error (MSE) between the model predictions and the actual data.

    The mean squared error (MSE) measures the average squared difference between the predicted 
    values and the actual values.

    Args:
        model (callable): The model function.
        params (array-like): The parameters of the model.
        x_data (array-like): The independent variable data.
        y_data (array-like): The observed dependent variable data.

    Returns:
        float: The mean squared error (MSE).
    """
    # Calculate the residuals
    residuals = calculate_residuals(model, params, xdata, ydata)

    # Calculate the mean squared error
    mse = mean_squared_error(ydata, model(xdata, *params))

    return mse


def calculate_bic(model, params, xdata, ydata):
    """
    Calculate the Bayesian Information Criterion (BIC).

    BIC is a criterion for model selection among a finite set of models.
    It provides a trade-off between goodness of fit and model complexity, penalizing models with more parameters.

    BIC is defined as:

    BIC = log(n) * k + n * log(RSS / n)

    where:
    - n is the number of data points.
    - k is the number of model parameters.
    - RSS is the residual sum of squares, calculated as the sum of squared differences between observed data and fitted data.

    A lower BIC value indicates a better trade-off between model fit and complexity.
    BIC can be used to compare models with different numbers of parameters and select the most appropriate model.

      Args:
        model (callable): The model function.
        params (array-like): The parameters of the model.
        x_data (array-like): The independent variable data.
        y_data (array-like): The observed dependent variable data.

    Returns:
        float: The Bayesian Information Criterion (BIC).
    """
    # Calculate the number of parameters in the model
    k = len(params)

    # Calculate the number of data points
    n = len(xdata)

    # Calculate the residual sum of squares
    rss = np.sum(np.square(calculate_residuals(model, params, xdata, ydata)))

    # Calculate the BIC using the formula BIC = n * log(RSS / n) + k * log(n), where n is the number of data points
    # and k is the number of parameters
    bic = n * np.log(rss / n) + k * np.log(n)

    return bic

def model_selection_criteria(model, xdata, ydata, p0=None, bounds=None):
    """
    Compute model selection criteria (AIC and BIC) for the given model.

    This function fits the specified model to the provided data using nonlinear least squares
    and calculates the Akaike Information Criterion (AIC) and Bayesian Information Criterion (BIC).

    Args:
        model (callable): A callable function representing the model to be fitted.
        xdata (array-like): Independent variable data.
        ydata (array-like): Dependent variable data.
        p0 (array-like, optional): Initial guess for the parameters of the model.
        bounds (2-tuple of array-like, optional): Bounds on parameters for the curve fit.

    Returns:
        tuple: A tuple containing the optimal parameters of the model, the covariance matrix, AIC, and BIC.
    """

def plot_model_fit(xdata, ydata, yfit, model_name):
    """
    Plot the observed data and the fitted curve for visualization.

    Args:
        xdata (array-like): Independent variable data.
        ydata (array-like): Observed dependent variable data.
        yfit (array-like): Fitted dependent variable data.
        model_name (str): Name of the fitted model.
    """


def aic(ydata, yfit, k):
    """
    Calculate the Akaike Information Criterion (AIC).

    AIC is a criterion for model selection among a finite set of models.
    It provides a trade-off between goodness of fit and model complexity, penalizing models with more parameters.

    AIC is defined as:

    AIC = n * log(RSS / n) + 2 * k

    where:
    - n is the number of data points.
    - k is the number of model parameters.
    - RSS is the residual sum of squares, calculated as the sum of squared differences between observed data and fitted data.

    A lower AIC value indicates a better trade-off between model fit and complexity.
    AIC can be used to compare models with different numbers of parameters and select the most appropriate model.

    Args:
        ydata (array-like): Observed data.
        yfit (array-like): Fitted data.
        k (int): Number of model parameters.

    Returns:
        float: AIC value.
    """
