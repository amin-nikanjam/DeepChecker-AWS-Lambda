import os
import sys
import random
from collections.abc import Iterable
from pathlib import Path
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 
from tensorflow.keras.layers import MaxPooling2D, Flatten
import callbacks as callbacks
import settings as settings
import metrics as metrics
import utils as utils
from utils import readable
from metadata import DNNState, InputData
from settings import CLASSIFICATION_KEY, REGRESSION_KEY

class DeepChecker:

    def __init__(self, name, data, model, app_path=None, buffer_scale=10):
        app_path = Path.cwd() if app_path == None else app_path
        log_fpath = settings.build_log_file_path(app_path, name)
        self.logger = settings.file_logger(log_fpath, name)
        config_fpath = settings.load_user_config_if_exists(app_path)
        self.config = settings.Config(config_fpath)
        inputs_data = InputData(data, model.problem_type)
        self.buffer_scale = buffer_scale
        self.pre_check = PreCheck(model, inputs_data, self.logger, self.config.pre_check)
        self.overfit_check = OverfitCheck(model, inputs_data, self.logger, self.config.overfit_check, self.buffer_scale)

    def setup(self, fixed_seed=True, tf_seed=42, np_seed=42, python_seed=42, use_multi_cores=False, use_GPU=False):
        if fixed_seed:
            if os.environ.get("PYTHONHASHSEED") != "0":
                print(
                    'You must set PYTHONHASHSEED=0 when running the python script If you wanna get reproducible results.')
            tf.random.set_seed(tf_seed)
            np.random.seed(np_seed)
            random.seed(python_seed)
     
        if not use_GPU:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    def run_full_checks(self,
                        overfit_batch=50,
                        overfit_iters=200,
                        post_fitness_batch=64,
                        post_fitness_epochs=301,
                        fixed_seed=True,
                        use_multi_cores=False,
                        use_GPU=True,
                        implemented_ops=[]):
        print('Setup...')
        self.setup(fixed_seed=fixed_seed, use_multi_cores=use_multi_cores, use_GPU=use_GPU)
        print('The checking process will begin, please wait few moments until the report is generated')
        self.run_pre_checks(overfit_batch, implemented_ops)
        # print('Prechecks were done successfully, now it"'"s time for overfit checks")
        self.run_overfit_checks(overfit_batch, overfit_iters)
        # self.run_post_checks(post_fitness_batch, post_fitness_epochs)
        print('Your model was tested successfully, Check the log file to find the full report...')

    def run_pre_checks(self, batch_size, implemented_ops):
        self.pre_check.run(batch_size, implemented_ops)
    # def run_post_checks(self, post_fitness_batch, post_fitness_epochs):
    #     self.post_check.run(post_fitness_batch, post_fitness_epochs)
    def run_overfit_checks(self, overfit_batch, overfit_iters):
        self.overfit_check.run(overfit_batch, overfit_iters)
    
    

class PreCheck:

    def __init__(self, model, inputs_data, main_logger, config):
        self.model = model
        self.inputs_data = inputs_data
        self.config = config
        self.main_msgs = settings.load_messages()
        self.main_logger = main_logger

    def react(self, message):
        if self.config.fail_on:
            raise Exception(message)
        else:
            self.main_logger.warning(message)


    """
    This function below checks the input data (features) if it contains any error.
    It stores lists of maximums, minimums, averages and standard deviations according to the type of data (homogenous/heterogenous).
    It verifies whether the features are normalized or not. It also detects constant features.
    """

    def _pre_check_features(self):
        if self.config.data.disabled: return
        if self.inputs_data.homogeneous:
            mas = [self.inputs_data.features_metadata['max']]
            mis = [self.inputs_data.features_metadata['min']]
            avgs = [self.inputs_data.features_metadata['mean']]
            stds = [self.inputs_data.features_metadata['std']]
        elif isinstance(self.inputs_data.features_metadata['max'], Iterable):
            mas = list(self.inputs_data.features_metadata['max'])
            mis = list(self.inputs_data.features_metadata['min'])
            avgs = list(self.inputs_data.features_metadata['mean'])
            stds = list(self.inputs_data.features_metadata['std'])
        else:
            mas = [self.inputs_data.features_metadata['max']]
            mis = [self.inputs_data.features_metadata['min']]
            avgs = [self.inputs_data.features_metadata['mean']]
            stds = [self.inputs_data.features_metadata['std']]
        for idx in range(len(mas)):
            if stds[idx] == 0.0:
                msg = self.main_msgs['features_constant'] if len(mas) == 1 else self.main_msgs['feature_constant'].format(idx)
                self.react(msg)
            elif any([utils.almost_equal(mas[idx], data_max) for data_max in self.config.data.normalized_data_maxs]) and \
                    any([utils.almost_equal(mis[idx], data_min) for data_min in self.config.data.normalized_data_mins]):
                return
            elif not (utils.almost_equal(stds[idx], 1.0) and utils.almost_equal(avgs[idx], 0.0)):
                msg = self.main_msgs['features_unnormalized'] if len(mas) == 1 else self.main_msgs['feature_unnormalized'].format(idx)
                self.react(msg)


    '''
    This function performs checks on output data (targets).
    Classification: It verifies if labels are balanced or not through comparing label frequencies to a predefined threshold.
    Regression: It stores lists of maximums, minimums, averages and standard deviations. Then, it verifies whether the targets are normalized or not.
    '''


    def _pre_check_targets(self):
        if self.config.data.disabled: return
        if self.inputs_data.problem_type == CLASSIFICATION_KEY:
            if self.inputs_data.targets_metadata['balance'] < self.config.data.labels_perp_min_thresh:
                self.react(self.main_msgs['unbalanced_labels'])
        elif self.inputs_data.problem_type == REGRESSION_KEY:
            if self.inputs_data.targets_metadata['count'] == 1:
                mas = [self.inputs_data.targets_metadata['max']]
                mis = [self.inputs_data.targets_metadata['min']]
                avgs = [self.inputs_data.targets_metadata['mean']]
                stds = [self.inputs_data.targets_metadata['std']]
            else:
                mas = list(self.inputs_data.targets_metadata['max'])
                mis = list(self.inputs_data.targets_metadata['min'])
                avgs = list(self.inputs_data.targets_metadata['mean'])
                stds = list(self.inputs_data.targets_metadata['std'])
            for idx in range(len(mas)):
                if utils.almost_equal(stds[idx], 0.0):
                    msg = self.main_msgs['targets_unnormalized'] if len(mas) == 1 else self.main_msgs['target_unnormalized'].format(idx)
                    self.react(msg)
                elif any([utils.almost_equal(mas[idx], data_max) for data_max in
                          self.config.data.normalized_data_maxs]) and \
                        any([utils.almost_equal(mis[idx], data_min) for data_min in self.config.data.normalized_data_mins]):
                    return
                elif not (utils.almost_equal(stds[idx], 1.0) and utils.almost_equal(avgs[idx], 0.0)):
                    msg = self.main_msgs['targets_unnormalized'] if len(mas) == 1 else self.main_msgs['target_unnormalized'].format(idx)
                    self.react(msg)


    '''This function performs checks on initial weights (before training). It verifies if the weights are poorly intialized (variance = 0).
    It uses the f_test to check the equality between the actual and recommended variance of weights depending on the used activation function:
    Better to use He initilization for weights with relu as an activation function
    Better to use Glorot/Xavier initialization for weights with tanh as an activation function
    Better to use LeCun initialization for weights with sigmoid as an activation function
    PS: If none of the mentioned metrics above is applied, then it is recommended to use a well-known intitalization for weights.
    '''


    def _pre_check_weights(self):
        if self.config.init_w.disabled: return
        model = self.model.build(self.model.n_classes)
        initial_weights = dict()
        weights = []
        names = []
        for layer in model.layers:
            if len(layer.weights)!=0 :
                weights.append(layer.weights[0].numpy())
                names.append(layer.weights[0].name)
        for i in range(0, len(weights)):
            initial_weights[names[i]] = weights[i]
        activations = []
        ch = ''
        activation_name = ''
        for layer in model.layers:
            if len(layer.weights)!=0:
                ch = str(layer.activation)
                activation_name = ch.split(' ')[1]
                activations.append(activation_name)
        i = 0
        for weight_name, weight_array in initial_weights.items():
            shape = weight_array.shape
            if len(shape) == 1 and shape[0] == 1: continue
            if utils.almost_equal(np.var(weight_array), 0.0, rtol=1e-8):
                self.react(self.main_msgs['poor_init'].format(weight_name))
            else:
                if len(shape) == 2:
                    fan_in = shape[0]
                    fan_out = shape[1]
                else:
                    receptive_field_size = np.prod(shape[:-2])
                    fan_in = shape[-2] * receptive_field_size
                    fan_out = shape[-1] * receptive_field_size
                lecun_F, lecun_test = metrics.pure_f_test(weight_array, np.sqrt(1.0 / fan_in),
                                                          self.config.init_w.f_test_alpha)
                he_F, he_test = metrics.pure_f_test(weight_array, np.sqrt(2.0 / fan_in),
                                                    self.config.init_w.f_test_alpha)
                glorot_F, glorot_test = metrics.pure_f_test(weight_array, np.sqrt(2.0 / (fan_in + fan_out)),
                                                            self.config.init_w.f_test_alpha)
                if activations[i] == 'relu' and not he_test:
                    abs_std_err = np.abs(np.std(weight_array) - np.sqrt(1.0 / fan_in))
                    i += 1
                    self.react(self.main_msgs['need_he'].format(weight_name, abs_std_err))
                elif activations[i] == 'tanh' and not glorot_test:
                    abs_std_err = np.abs(np.std(weight_array) - np.sqrt(2.0 / fan_in))
                    i += 1
                    self.react(self.main_msgs['need_glorot'].format(weight_name, abs_std_err))
                elif activations[i] == 'sigmoid' and not lecun_test:
                    abs_std_err = np.abs(np.std(weight_array) - np.sqrt(2.0 / (fan_in + fan_out)))
                    i += 1
                    self.react(self.main_msgs['need_lecun'].format(weight_name, abs_std_err))
                elif not (lecun_test or he_test or glorot_test):
                    self.react(self.main_msgs['need_init_well'].format(weight_name))
                else:
                    i += 1


    '''
    This function below performs checks on initial biases (before training). It verifies biases exist and are initialized to 0.
    Classification: It tests if the biases of the last layer are not equal to zero in case of unbalanced data, and if they match the ratio of labels.
    Regression: It tests if the biases of the last layer are equal to the mean value.
    '''

    def _pre_check_biases(self):
        if self.config.init_b.disabled: return
        model = self.model.build(self.model.n_classes)
        initial_biases = dict()
        biases = []
        names = []
        for layer in model.layers:
            if len(layer.weights)!=0:
                biases.append(layer.weights[1].numpy())
                names.append(layer.weights[1].name)
        for i in range(0, len(biases)):
            initial_biases[names[i]] = biases[i]
        if not(initial_biases):
            self.react(self.main_msgs['need_bias'])
        else:
            checks = []
            for b_name, b_array in initial_biases.items():
                checks.append(np.sum(b_array)==0.0)
            if self.inputs_data.problem_type == CLASSIFICATION_KEY and \
                self.inputs_data.targets_metadata['balance'] < self.config.data.labels_perp_min_thresh:
                if checks[-1]:
                    self.react(self.main_msgs['last_bias'])
                elif not checks[-1]:
                    bias_indices = np.argsort(b_array)
                    probas_indices = np.argsort(self.inputs_data.targets_metadata['probas'])
                    if not np.equal(bias_indices, probas_indices):
                        self.react(self.main_msgs['ineff_bias_cls'])
            elif self.inputs_data.problem_type == REGRESSION_KEY:
                if self.inputs_data.targets_metadata['count'] == 1:
                    avgs = [self.inputs_data.targets_metadata['mean']]
                    stds = [self.inputs_data.targets_metadata['std']]
                else:
                    avgs = list(self.inputs_data.targets_metadata['mean'])
                    stds = list(self.inputs_data.targets_metadata['std'])
                var_coefs = [std/avg for avg, std in zip(avgs, stds)]
                low_var_coefs_indices = [i for i, var_coef in enumerate(var_coefs) if var_coef <= 1e-3]
                for idx in low_var_coefs_indices:
                    b_value = float(b_array[idx])
                    if not(utils.almost_equal(b_value, avgs[idx])):
                        self.react(self.main_msgs['ineff_bias_regr'].format(idx))
            elif not np.all(checks):
                self.react(self.main_msgs['zero_bias'])


    '''
    This function below performs checks on loss.
    Following the equality check, It checks if the loss is decreasing compared to a predefined growth rate
    Through this function, the model is trained and the loss error (expected loss - initial loss) is generated, through which a poor loss initialization is detected.
    '''

    def _pre_check_loss(self):
        if self.config.init_loss.disabled: return
        model = self.model.build(self.model.n_classes)
        batch_x, batch_y = self.inputs_data.get_sample(self.config.init_loss.sample_size)
        dbatch_x = []
        dbatch_y = []
        losses = []
        n = self.config.init_loss.size_growth_rate
        while n <= (self.config.init_loss.size_growth_rate * self.config.init_loss.size_growth_iters):
            derived_batch_x  = np.concatenate(n*[batch_x], axis=0)
            derived_batch_y = np.concatenate(n*[batch_y], axis=0)
            for element in derived_batch_x:
                dbatch_x.append(element)
            for element in derived_batch_y:
                dbatch_y.append(element)
            dbx = tf.convert_to_tensor(dbatch_x)
            dby = tf.convert_to_tensor(dbatch_y)
            with tf.GradientTape() as tape:
                model.compile(loss = self.model.loss_fct, optimizer = self.model.optimizer)
                history = model.fit(dbx,dby,verbose=0)
                losses.append(history.history['loss'][0])
                n *= self.config.init_loss.size_growth_rate
        rounded_loss_rates = [round(losses[i + 1] / losses[i]) for i in range(len(losses) - 1)]
        equality_checks = sum(
            [(loss_rate == self.config.init_loss.size_growth_rate) for loss_rate in rounded_loss_rates])
        if equality_checks == len(rounded_loss_rates):
            self.react(self.main_msgs['poor_reduction_loss'])
        if self.model.problem_type == CLASSIFICATION_KEY:
            with tf.GradientTape() as tape:
                model.compile(loss=self.model.loss_fct, optimizer=self.model.optimizer)
                history_all = model.fit(batch_x, batch_y, verbose=0)
                initial_loss = (history_all.history['loss'][0])
                expected_loss = -np.log(1 / self.inputs_data.targets_metadata['labels'])
                err = np.abs(initial_loss - expected_loss)
                if err >= self.config.init_loss.dev_ratio * expected_loss:
                    self.react(self.main_msgs['poor_init_loss'].format(readable(err / expected_loss)))


    '''
    This function below performs checks on gradients.
    It calculates theoretical and numerical gradients of loss with respect to initial weights.
    Then, it determines the relative error between both gradients, that should not exceed a predefined threshold.
    '''

    def _pre_check_gradients(self):
        if self.config.grad.disabled: return
        def intermediate_function(x):
            return tf.convert_to_tensor(loss_value)
        model = self.model.build(self.model.n_classes)
        model.compile(loss=self.model.loss_fct, optimizer=self.model.optimizer, run_eagerly=True)
        for _ in range(self.config.grad.warm_up_steps):
            batch_x, batch_y = self.inputs_data.get_sample(self.config.grad.warm_up_batch)
            history = model.fit(batch_x, batch_y, verbose=0)
        loss_value = history.history['loss'][0]
        weights = []
        for layer in model.layers:
            if len(layer.weights)!=0:
                weights.append(layer.weights[0])
        for i in range(len(weights)):
            theoretical, numerical = tf.test.compute_gradient(
                intermediate_function,
                x=[weights[i]],
                delta=self.config.grad.delta)
            theoretical, numerical = theoretical[0].flatten(), numerical[0].flatten()
            total_dims, sample_dims = len(theoretical), int(self.config.grad.ratio_of_dimensions*len(theoretical))
            indices = np.random.choice(np.arange(total_dims), sample_dims, replace=False)
            theoretical_sample = theoretical[indices]
            numerical_sample = numerical[indices]
            numerator = np.linalg.norm(theoretical_sample - numerical_sample)
            denominator = np.linalg.norm(theoretical_sample) + np.linalg.norm(numerical_sample)
            if denominator == 0.0:
                print('Unable to calculate the relative error between theoretical and numerical gradient: Gradients are equal to 0.\nThis checker will be ignored.')
                return
            relerr = numerator / denominator
            if relerr > self.config.grad.relative_err_max_thresh:
                self.react(self.main_msgs['grad_err'].format(all_weights[i], readable(relerr), self.config.grad.relative_err_max_thresh))


    '''
    This function below performs checks on the fitting data step. It tests if the DNN model is fitting well a sample batch of data during training:
    It detects if there is an underfitting problem based on the loss value (Nan or +/- Inf). Similarly, it detects the same problem while feeding the model iteratively if:
    Classification: 1 - max(accuracies) > predefined threshold
    Regression: min(accuracies) > predefined threshold
    This function also demonstrates if there is a data dependency problem.
    '''


    def _pre_check_fitting_data_capability(self):
        if self.config.prop_fit.disabled: return
        def _loss_is_stable(loss_value):
            if np.isnan(loss_value):
                self.react(self.main_msgs['nan_loss'])
                return False
            if np.isinf(loss_value):
                self.react(self.main_msgs['inf_loss'])
                return False
            return True
        model = self.model.build(self.model.n_classes)
        batch_x, batch_y = self.inputs_data.get_sample(self.config.prop_fit.single_batch_size)
        real_losses = []
        real_accs = []
        model.compile(loss=self.model.loss_fct, optimizer=self.model.optimizer, metrics=[self.model.perf])
        for _ in range(self.config.prop_fit.total_iters):
            history = model.fit(batch_x, batch_y, verbose=0)
            real_loss = history.history['loss'][0]
            real_acc = history.history['accuracy'][0]
            real_losses.append(real_loss)
            real_accs.append(real_acc)
            if not(_loss_is_stable(real_loss)):
                self.react(self.main_msgs['underfitting_single_batch'])
                return
        loss, acc = (real_losses[-1] + self.model.reg_loss, real_accs[-1]) if self.model.reg_loss!= None else (real_losses[-1], real_accs[-1])
        underfitting_prob = False
        if self.inputs_data.problem_type == CLASSIFICATION_KEY:
            if 1.0 - max(real_accs) > self.config.prop_fit.mislabeled_rate_max_thresh:
                self.react(self.main_msgs['underfitting_single_batch'])
                underfitting_prob = True
        elif self.inputs_data.problem_type == REGRESSION_KEY:
            if min(real_accs) > self.config.prop_fit.mean_error_max_thresh:
                self.react(self.main_msgs['underfitting_single_batch'])
                underfitting_prob = True
        loss_smoothness = metrics.smoothness(np.array(real_losses))
        min_loss = np.min(np.array(real_losses))
        if min_loss <= self.config.prop_fit.abs_loss_min_thresh or (min_loss <= self.config.prop_fit.loss_min_thresh and loss_smoothness > self.config.prop_fit.smoothness_max_thresh):
            self.react(self.main_msgs['zero_loss']); return
        if not (underfitting_prob): return
        zeroed_batch_x = np.zeros_like(batch_x)
        fake_losses = []
        model.compile(loss = self.model.loss_fct, optimizer = self.model.optimizer)
        for _ in range(self.config.prop_fit.total_iters):
            history = model.fit(zeroed_batch_x, batch_y, verbose=0)
            fake_loss = history.history['loss'][0]
            fake_losses.append(fake_loss)
            if not (_loss_is_stable(fake_loss)): return
        stability_test = np.array([_loss_is_stable(loss_value) for loss_value in (real_losses + fake_losses)])
        if (stability_test == False).any():
            last_real_losses = real_losses[-self.config.prop_fit.sample_size_of_losses:]
            last_fake_losses = fake_losses[-self.config.prop_fit.sample_size_of_losses:]
            if not (metrics.are_significantly_different(last_real_losses, last_fake_losses)):
                self.react(self.main_msgs['data_dep'])

    def run(self, batch_size, implemented_ops):
        if self.config.disabled: return
        #print('The PreCheck phase is starting. Checking features (inputs)...\n')
        self._pre_check_features()
        #print('Features --> Checked! \nChecking targets (outputs)...\n')
        self._pre_check_targets()
        # print('Targets --> Checked! \nChecking weights...\n')
        self._pre_check_weights()
        # print('Weights --> Checked! \nChecking biases...\n')
        self._pre_check_biases()
        # print('Biases --> Checked! \nChecking losses...\n')
        self._pre_check_loss()
        # print('Losses --> Checked! \nChecking gradients...\n')
        self._pre_check_gradients()
        # print('Gradients --> Checked! \nChecking fitting data capability...\n')
        #self._pre_check_fitting_data_capability()
        #print('fitting data capability --> Checked! \nEnd of PreChecks!\n')



class OverfitCheck:

    def __init__(self, model, inputs_data, logger, config, buffer_scale):
        self.model = model
        self.logger = logger
        self.config = config
        self.inputs_data = inputs_data
        self.buffer_scale = buffer_scale

    def build_callbacks(self, model, overfit_batch):
        overfit_callbacks = [
        callbacks.OverfitWeightCallback(model=model, main_logger=self.logger, config=self.config.weight, buffer_scale= self.buffer_scale, fail_on=self.config.fail_on),
        callbacks.OverfitBiasCallback(model=model, main_logger=self.logger, config=self.config.bias, buffer_scale=self.buffer_scale, fail_on=self.config.fail_on),
        callbacks.OverfitActivationCallback(inputs_data=self.inputs_data, overfit_batch = overfit_batch, model=model, main_logger=self.logger, \
                                            config=self.config.act, buffer_scale=self.buffer_scale, fail_on=self.config.fail_on),
        callbacks.OverfitGradientCallback(loss_fct=self.model.loss_fct, inputs_data=self.inputs_data, overfit_batch=overfit_batch, \
                                        model=model, main_logger=self.logger, config=self.config.grad, buffer_scale = self.buffer_scale, fail_on=self.config.fail_on),
        callbacks.OverfitLossCallback(loss_fct=self.model.loss_fct, perf=self.model.perf, inputs_data=self.inputs_data, overfit_batch=overfit_batch, \
                                      model=model, main_logger=self.logger, config=self.config.loss, buffer_scale = self.buffer_scale, fail_on=self.config.fail_on)
    ]
        return overfit_callbacks

    def run(self, overfit_batch, overfit_iters):
        if self.config.disabled: return
        model = self.model.build()
        model.compile(loss = self.model.loss_fct, optimizer = self.model.optimizer, metrics = [self.model.perf])
        self.callbacks = self.build_callbacks(model, overfit_batch)
        batch_x, batch_y = self.inputs_data.get_sample(overfit_batch)
        patience = self.config.patience
        for i in range(overfit_iters):
            history = model.fit(batch_x, batch_y, callbacks=self.callbacks, verbose=0)
            perf_metric = history.history[self.model.perf.name][0]
            if self.model.problem_type == CLASSIFICATION_KEY and perf_metric == self.config.classif_perf_thresh:
                patience -= 1
            elif self.model.problem_type == REGRESSION_KEY and perf_metric < self.config.regr_perf_thresh:
                patience -= 1
            else:
                patience = self.config.patience
            if patience == 0: break



        


