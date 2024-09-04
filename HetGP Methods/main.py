import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from uci_datasets import Dataset
import gpflow as gpf
import tensorflow_probability as tfp
import evidential_deep_learning as edl
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm



#-------------------------Metrics-------------------------------------#
def ece(y_test, preds, strategy='uniform'):
    y_test = y_test.flatten() if isinstance(y_test, np.ndarray) else y_test.numpy().flatten()
    preds = preds.flatten() if isinstance(preds, np.ndarray) else preds.numpy().flatten()
    
    if len(y_test) != len(preds):
        raise ValueError(f"Shapes of y_test and preds must match. Got shapes {y_test.shape} and {preds.shape}.")
        
    df = pd.DataFrame({'target': y_test, 'proba': preds})
    if strategy == 'uniform':
        bins = np.linspace(0, 1, 11)
        df['bin'] = np.digitize(df['proba'], bins) - 1

    df_bin_groups = df.groupby('bin').agg({
        'target': 'mean',
        'proba': 'mean',
        'bin': 'count'
    }).rename(columns={'bin': 'count'})

    df_bin_groups['ece'] = (df_bin_groups['target'] - df_bin_groups['proba']).abs() * (df_bin_groups['count'] / df.shape[0])
    return df_bin_groups['ece'].sum()


def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

def nlpd(y_true, mu, sigma):
    nlpd = np.mean(0.5 * np.log(2 * np.pi * sigma**2) + 0.5 * ((y_true - mu) / sigma)**2)
    return -nlpd

#-------------------------------Dataset Functions---------------------------------------#

def load_and_preprocess_data(dataset_name):
    data = Dataset(dataset_name)
    X_train, y_train, X_test, y_test = data.get_split(split=0)
    
    X_train_array, X_test_array = np.asarray(X_train), np.asarray(X_test)
    y_train_array, y_test_array = np.asarray(y_train), np.asarray(y_test)
    
    train_indices = np.random.permutation(X_train_array.shape[0])
    X_train_array = X_train_array[train_indices]
    y_train_array = y_train_array[train_indices]
    scaler = MinMaxScaler()
    
    X_train_scaled = scaler.fit_transform(X_train_array)
    X_test_scaled = scaler.transform(X_test_array)
    y_train_scaled = scaler.fit_transform(y_train_array.reshape(-1, 1))
    y_test_scaled = scaler.transform(y_test_array.reshape(-1, 1))
    
    return X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled




#-------------------------------Model Functions---------------------------------------#
def build_nn_model(input_dim):
    inputs = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(500, activation='relu')(inputs)
    x = tf.keras.layers.Dense(500, activation='relu')(x)
    x = tf.keras.layers.Dense(500, activation='relu')(x)
    output = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_nn_model(model, X_train_scaled, y_train_scaled, epochs=100, batch_size=100):
    model.fit(X_train_scaled, y_train_scaled, batch_size=batch_size, epochs=epochs)
    return model

def build_gp_model(kernel, likelihood, inducing_variable, num_latent_gps):
    return gpf.models.SVGP(
        kernel=kernel,
        likelihood=likelihood,
        inducing_variable=inducing_variable,
        num_latent_gps=num_latent_gps
    )

def train_gp_model(model, data, epochs=1000, log_freq=100):
    loss_fn = model.training_loss_closure(data)
    gpf.utilities.set_trainable(model.q_mu, False)
    gpf.utilities.set_trainable(model.q_sqrt, False)

    variational_vars = [(model.q_mu, model.q_sqrt)]
    natgrad_opt = gpf.optimizers.NaturalGradient(gamma=0.1)  
    adam_opt = tf.optimizers.Adam(0.01)

    @tf.function
    def optimisation_step():
        natgrad_opt.minimize(loss_fn, variational_vars)
        adam_opt.minimize(loss_fn, model.trainable_variables)

    for epoch in range(1, epochs + 1):
        try:
            optimisation_step()
        except tf.errors.InvalidArgumentError as e:
            print(f"NaN or Inf error at epoch {epoch}: {e}")
            break
        if epoch % log_freq == 0:
            print(f"Epoch {epoch} - Loss: {loss_fn().numpy() : .4f}")

    return model

def train_gp_model_hom(model, data, epochs=1000, log_freq=100):
    loss_fn = model.training_loss_closure(data)
    gpf.utilities.set_trainable(model.q_mu, False)
    gpf.utilities.set_trainable(model.q_sqrt, False)

    variational_vars = [(model.q_mu, model.q_sqrt)]
    natgrad_opt = gpf.optimizers.NaturalGradient(gamma=0.1)
    adam_opt = tf.optimizers.Adam(0.01) 

    @tf.function
    def optimisation_step():
        natgrad_opt.minimize(loss_fn, variational_vars)
        adam_opt.minimize(loss_fn, model.trainable_variables)

    for epoch in range(1, epochs + 1):
        try:
            optimisation_step()
        except tf.errors.InvalidArgumentError as e:
            print(f"NaN or Inf error at epoch {epoch}: {e}")
            break
        if epoch % log_freq == 0:
            print(f"Epoch {epoch} - Loss: {loss_fn().numpy() : .4f}")

    return model


def build_evidential_model(input_dim):
    inputs = tf.keras.Input(input_dim)
    x = tf.keras.layers.Dense(500, activation='relu')(inputs)
    x = tf.keras.layers.Dense(500, activation='relu')(x)
    x = tf.keras.layers.Dense(500, activation='relu')(x)
    output = edl.layers.DenseNormalGamma(1)(x)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss=lambda true, pred: edl.losses.EvidentialRegression(true, pred, coeff=1e-2))
    return model

def train_evidential_model(model, X_train_scaled, y_train_scaled, epochs=100, batch_size=100):
    model.fit(X_train_scaled, y_train_scaled, batch_size=batch_size, epochs=epochs)
    return model

def evaluate_evidential_model(model, X_test_scaled, y_test_scaled):
    pred = model.predict(X_test_scaled)
    mu_edl, v_edl, alpha, beta = tf.split(pred, 4, axis=-1)
    mu_edl, var_edl = mu_edl[:, 0], np.sqrt(beta / (v_edl * (alpha - 1)))[:, 0]
    
    #y_test_scaled = y_test_scaled.flatten()
    mu_edl, var_edl = mu_edl.numpy().reshape(-1, 1), np.minimum(var_edl, 1e3).reshape(-1, 1)
    ale_edl =  np.sqrt(beta / ((alpha - 1)))
    ale_edl = np.minimum(ale_edl, 1e3)[:, 0]

    results = {
        "rmse": rmse(y_test_scaled, mu_edl).numpy(),
        "ece": ece(y_test_scaled, mu_edl),
        "nlpd": nlpd(y_test_scaled, mu_edl, (var_edl+ale_edl))
    }
    
    return results

#-------------------------------Main Function---------------------------------------#
def run_all(dataset_name):

    X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled = load_and_preprocess_data(dataset_name)
    
    # Neural Network Model
    nn_model = build_nn_model(X_train_scaled.shape[1])
    nn_model = train_nn_model(nn_model, X_train_scaled, y_train_scaled)
    nn_mu_train = nn_model.predict(X_train_scaled)
    nn_mu_test = nn_model.predict(X_test_scaled)
    
    train_rmse = rmse(y_train_scaled, nn_mu_train)
    test_rmse = rmse(y_test_scaled, nn_mu_test)
    test_ece = ece(y_test_scaled, nn_mu_test)
    
    residuals = y_train_scaled - nn_mu_train
    
    # Heteroscedastic GP Model
    likelihood_hetero = gpf.likelihoods.HeteroskedasticTFPConditional(
        distribution_class=tfp.distributions.Normal,
        scale_transform=tfp.bijectors.Exp()
    )
    kernel_hetero = gpf.kernels.SeparateIndependent([gpf.kernels.SquaredExponential(), gpf.kernels.SquaredExponential()])
    Z_hetero = np.linspace(-1, 1, 50 * X_train_scaled.shape[1])[:, None].reshape(-1, X_train_scaled.shape[1])
    inducing_variable_hetero = gpf.inducing_variables.SeparateIndependentInducingVariables(
        [gpf.inducing_variables.InducingPoints(Z_hetero), gpf.inducing_variables.InducingPoints(Z_hetero)]
    )
    gp_model_hetero = build_gp_model(kernel_hetero, likelihood_hetero, inducing_variable_hetero, 2)
    gp_model_hetero = train_gp_model(gp_model_hetero, (X_train_scaled, residuals))
    mu_res_hetero, var_res_hetero = gp_model_hetero.predict_y(X_test_scaled)
    mu_res_f_hetero, var_res_f_hetero = gp_model_hetero.predict_f(X_test_scaled)
    
    het_rmse = rmse(y_test_scaled.flatten(), nn_mu_test.flatten() + mu_res_hetero.numpy().flatten())
    ece_score_hetero = ece(y_test_scaled.flatten(), nn_mu_test.flatten() + mu_res_hetero.numpy().flatten())
    nlpd_score_hetero = nlpd(y_test_scaled.flatten(), nn_mu_test.flatten() + mu_res_hetero, np.sqrt(var_res_hetero + var_res_f_hetero[:,0]+var_res_f_hetero[:,1]))
    
    results_hetero = {
        "train_rmse_hetero": het_rmse.numpy(),
        #"test_rmse_hetero": test_rmse.numpy(),
        "ece_hetero": ece_score_hetero,
        "nlpd_hetero": nlpd_score_hetero,
    }

    # Homoscedastic GP Model
    likelihood_homo = gpf.likelihoods.Gaussian()
    kernel_homo = gpf.kernels.SquaredExponential()
    Z_homo = np.linspace(-1, 1, 50 * X_train_scaled.shape[1])[:, None].reshape(-1, X_train_scaled.shape[1])
    inducing_variable_homo = gpf.inducing_variables.InducingPoints(Z_homo)
    gp_model_homo = build_gp_model(kernel_homo, likelihood_homo, inducing_variable_homo, 1)
    gp_model_homo = train_gp_model_hom(gp_model_homo, (X_train_scaled, residuals))
    mu_res_homo, var_res_homo = gp_model_homo.predict_y(X_test_scaled)
    
    mu_res_f_homo, var_res_f_homo = gp_model_homo.predict_y(X_test_scaled)
    
    rmse_score_homo = rmse(y_test_scaled.flatten(), nn_mu_test.flatten() + mu_res_homo.numpy().flatten())
    ece_score_homo = ece(y_test_scaled.flatten(), nn_mu_test.flatten() + mu_res_homo.numpy().flatten())
    nlpd_score_homo = nlpd(y_test_scaled.flatten(), nn_mu_test.flatten() + mu_res_homo, np.sqrt(var_res_homo + var_res_f_homo))
    
    results_homo = {
        "train_rmse_homo": rmse_score_homo.numpy(),
        #"test_rmse_homo": test_rmse.numpy(),
        "ece_homo": ece_score_homo,
        "nlpd_homo": nlpd_score_homo,
    }

    # Evidential Model
    evidential_model = build_evidential_model(X_train_scaled.shape[1])
    evidential_model = train_evidential_model(evidential_model, X_train_scaled, y_train_scaled)
    results_evidential = evaluate_evidential_model(evidential_model, X_test_scaled, y_test_scaled)

    return {
        "nn_rmse": test_rmse,
        "nn_ece": test_ece,
        "hetero": results_hetero,
        "homo": results_homo,
        "evidential": results_evidential
    }


all_results = []
for _ in tqdm(range(10)):
    results = run_all("autompg")
    all_results.append(results)

for i, result in enumerate(all_results):
    print(f"Iteration {i+1}: {result}")

