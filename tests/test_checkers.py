from collections.abc import Iterable
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

# Import modules from parent folder
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
 

import metrics
from metadata import InputData
from utils import readable
import data as data
import utils
import interfaceData
import settings

from settings import CLASSIFICATION_KEY, REGRESSION_KEY


def pre_check_features(inputs_data):
    app_path = Path.cwd()
    config_fpath = settings.load_user_config_if_exists(app_path)
    config = settings.Config(config_fpath).pre_check
    main_msgs = settings.load_messages()

 
    if inputs_data.homogeneous:
        mas = [inputs_data.features_metadata['max']]
        mis = [inputs_data.features_metadata['min']]
        avgs = [inputs_data.features_metadata['mean']]
        stds = [inputs_data.features_metadata['std']]
    if isinstance(inputs_data.features_metadata['max'], Iterable):
        mas = list(inputs_data.features_metadata['max'])
        mis = list(inputs_data.features_metadata['min'])
        avgs = list(inputs_data.features_metadata['mean'])
        stds = list(inputs_data.features_metadata['std'])
    else:
        mas = [inputs_data.features_metadata['max']]
        mis = [inputs_data.features_metadata['min']]
        avgs = [inputs_data.features_metadata['mean']]
        stds = [inputs_data.features_metadata['std']]

    for idx in range(len(mas)):
        # '''Data is constant'''
        if stds[idx] == 0.0:
            msg = main_msgs['features_constant'] if len(mas) == 1 else main_msgs['feature_constant'].format(
                idx)
            return msg
        # '''Data is normalized and not constant'''
        elif any([utils.almost_equal(mas[idx], data_max) for data_max in config.data.normalized_data_maxs]) and \
                any([utils.almost_equal(mis[idx], data_min) for data_min in config.data.normalized_data_mins]):
            msg = 'everything is clear'
            return msg
        # '''Data is unnormalized'''
        elif not (utils.almost_equal(stds[idx], 1.0) and utils.almost_equal(avgs[idx], 0.0)):
            msg = main_msgs['features_unnormalized'] if len(mas) == 1 else main_msgs[
                'feature_unnormalized'].format(idx)
            return msg



def test_features():
    '''Data used to show the message: everything is clear'''
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    '''Data used to show the message: Features are constant'''
    x_train1 = np.ones((28, 28), dtype=np.int64)
    x_test1 = np.ones((28, 28), dtype=np.int64)
    y_train1 = np.ones((28,28), dtype=np.int64)
    y_test1 = np.ones((28,28), dtype=np.int64)
    '''Data used to show the message: Features seem to be unnormalized'''
    x_train2 = np.random.randint(1, 4000, (60000, 28, 28), dtype=np.int64)
    x_test2 = np.random.randint(1, 4000, (10000, 28, 28), dtype=np.int64)
    y_train2 = np.random.randint(1, 4000, (60000,), dtype=np.int64)
    y_test2 = np.random.randint(1, 4000, (10000,), dtype=np.int64)

    trains = [[x_train, y_train, x_test, y_test], [x_train1, y_train1, x_test1, y_test1], [x_train2, y_train2, x_test2, y_test2]]
    ok = True
    for i in range(len(trains)):
        data_loader_under_test = data.DataLoaderFromArrays(trains[i][0], trains[i][1], shuffle=True, one_hot=True, normalization=ok)
        test_data_loader = data.DataLoaderFromArrays(trains[i][2], trains[i][3], shuffle=True, one_hot=True, normalization=ok)
        data_to_test = interfaceData.build_data_interface(data_loader_under_test, test_data_loader, homogeneous=True)
        inputs_data = InputData(data_to_test, 'classification')
    # # assert pre_check_features(inputs_data) == 'everything is clear'
        if i == 0:
            assert pre_check_features(inputs_data) == 'everything is clear'
            ok = False
        if i == 1:
            assert pre_check_features(inputs_data) == 'Features are constant'
        if i == 2:
            assert pre_check_features(inputs_data) == 'Features seem to be unnormalized'


def pre_check_targets(inputs_data):
    app_path = Path.cwd()
    config_fpath = settings.load_user_config_if_exists(app_path)
    config = settings.Config(config_fpath).pre_check
    main_msgs = settings.load_messages()
    if inputs_data.problem_type == CLASSIFICATION_KEY:
        if inputs_data.targets_metadata['balance'] < config.data.labels_perp_min_thresh:
            msg = main_msgs['unbalanced_labels']
            return msg
    elif inputs_data.problem_type == REGRESSION_KEY:
        if inputs_data.targets_metadata['count'] == 1:
            mas = [inputs_data.targets_metadata['max']]
            mis = [inputs_data.targets_metadata['min']]
            avgs = [inputs_data.targets_metadata['mean']]
            stds = [inputs_data.targets_metadata['std']]
        else:
            mas = list(inputs_data.targets_metadata['max'])
            mis = list(inputs_data.targets_metadata['min'])
            avgs = list(inputs_data.targets_metadata['mean'])
            stds = list(inputs_data.targets_metadata['std'])
        for idx in range(len(mas)):
            if utils.almost_equal(stds[idx], 0.0):
                msg = main_msgs['targets_constant'] if len(mas) == 1 else main_msgs['target_constant'].format(idx)
                return msg
            elif any([utils.almost_equal(mas[idx], data_max) for data_max in config.data.normalized_data_maxs]) and \
            any([utils.almost_equal(mis[idx], data_min) for data_min in config.data.normalized_data_mins]):
                msg = 'everything is clear'
                return msg
            elif not (utils.almost_equal(stds[idx], 1.0) and utils.almost_equal(avgs[idx], 0.0)):
                msg = main_msgs['targets_unnormalized'] if len(mas) == 1 else main_msgs['target_unnormalized'].format(idx)
                return msg
 
def test_targets():
    '''Data used to show the message: Groundtruth Labels are unbalanced, which requires adaptive algorithms'''
    x_train0 = np.ones((314, 9), dtype=np.int64)
    y = np.array([[0],[1]])
    y_train0 = np.repeat(y, [1, 313], axis=0)
    x_test0 = np.ones((78, 9), dtype=np.int64)
    y_test0 = np.repeat(y, [1, 77], axis=0)

    '''Data used to show the message: everything is clear'''
    dataset = pd.read_csv('tests/data/auto-mpg.csv')
    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)
    train_features = train_dataset.copy()
    test_features = test_dataset.copy()
    train_labels = train_features.pop('MPG')
    test_labels = test_features.pop('MPG')
    x_train, y_train = train_features.to_numpy(), train_labels.to_numpy().reshape(-1,1)
    x_test, y_test = test_features.to_numpy(), test_labels.to_numpy().reshape(-1,1)
    '''Data used to show the message: Outputs are constant'''
    x_train1 = np.ones((314, 9), dtype=np.int64)
    y_train1 = np.ones((314, 1), dtype=np.int64)
    x_test1 = np.ones((78, 9), dtype=np.int64)
    y_test1 = np.ones((78, 1), dtype=np.int64)

    '''Data used to show the message: Outputs seem to be unnormalized'''
    x_train2 = np.random.randint(0, 10000, (314, 9), dtype=np.int64)
    x_test2 = np.random.randint(0, 10000, (78, 9), dtype=np.int64)
    y_train2 = np.random.randint(0, 10000, (314, 1), dtype=np.int64)
    y_test2 = np.random.randint(0, 10000, (78, 1), dtype=np.int64)

    trains = [[x_train0, y_train0, x_test0, y_test0], [x_train, y_train, x_test, y_test], [x_train1, y_train1, x_test1, y_test1], [x_train2, y_train2, x_test2, y_test2]]
    ok = False
    problem_type = CLASSIFICATION_KEY
    for i in range(len(trains)):
        data_loader_under_test = data.DataLoaderFromArrays(trains[i][0], trains[i][1], problem_type=problem_type, shuffle=True, one_hot=False, target_scaling=ok)
        test_data_loader = data.DataLoaderFromArrays(trains[i][2], trains[i][3], problem_type=problem_type, shuffle=True, one_hot=False, target_scaling=ok)
        data_to_test = interfaceData.build_data_interface(data_loader_under_test, test_data_loader, homogeneous=True)
        inputs_data = InputData(data_to_test, problem_type)
        if i == 0:
            assert pre_check_targets(inputs_data) == 'Groundtruth Labels are unbalanced, which requires adaptive algorithms'
            problem_type = REGRESSION_KEY
            ok = True
        if i == 1:
            assert pre_check_targets(inputs_data) == 'everything is clear'
            ok = False
        if i == 2:
            assert pre_check_targets(inputs_data) == 'Outputs are constant'
        if i == 3:
            assert pre_check_targets(inputs_data) == 'Outputs seem to be unnormalized'



def pre_check_weights(weight_name, weight_array, activation):
    app_path = Path.cwd()
    config_fpath = settings.load_user_config_if_exists(app_path)
    config = settings.Config(config_fpath).pre_check
    main_msgs = settings.load_messages()
    shape = weight_array.shape
    if len(shape) == 1 and shape[0] == 1:
        return 'ignored because of the shape'
    if utils.almost_equal(np.var(weight_array), 0.0, rtol=1e-8):
        return main_msgs['poor_init'].format(weight_name)
    else:
        if len(shape) == 2:
            fan_in = shape[0]
            fan_out = shape[1]
        else:
            receptive_field_size = np.prod(shape[:-2])
            fan_in = shape[-2] * receptive_field_size
            fan_out = shape[-1] * receptive_field_size
        lecun_F, lecun_test = metrics.pure_f_test(weight_array, np.sqrt(1.0 / fan_in), config.init_w.f_test_alpha)
        he_F, he_test = metrics.pure_f_test(weight_array, np.sqrt(2.0 / fan_in), config.init_w.f_test_alpha)
        glorot_F, glorot_test = metrics.pure_f_test(weight_array, np.sqrt(2.0 / (fan_in + fan_out)), config.init_w.f_test_alpha)
        if activation == 'relu' and not he_test:
            abs_std_err = np.abs(np.std(weight_array) - np.sqrt(1.0 / fan_in))
            return main_msgs['need_he'].format(weight_name, abs_std_err)
        elif activation == 'tanh' and not glorot_test:
            abs_std_err = np.abs(np.std(weight_array) - np.sqrt(2.0 / fan_in))
            return main_msgs['need_glorot'].format(weight_name, abs_std_err)
        elif activation == 'sigmoid' and not lecun_test:
            abs_std_err = np.abs(np.std(weight_array) - np.sqrt(2.0 / (fan_in + fan_out)))
            return main_msgs['need_lecun'].format(weight_name, abs_std_err)
        elif not (lecun_test or he_test or glorot_test):
            return main_msgs['need_init_well'].format(weight_name)
        else:
            return 'everything is clear'



#
# name = 'dense/kernel:0'
# array = np.random.randint(1, 5, (64, 1), dtype=np.int64)
# activation = 'tanh'
# print(pre_check_weights(name, array, activation))

 
# def test_weights():
#     names = ['conv2_d/kernel:0', 'conv2_d/kernel:0', 'dense/kernel:0', 'dense/kernel:0', 'dense/kernel:0', 'dense/kernel:0', 'conv2_d/kernel:0']
#     arrays = [np.array([1]), np.ones((5,5,1,32), dtype=np.int64),
#               np.random.randint(1,5, (64,1), dtype = np.int64), np.random.randint(1,5, (64,1), dtype = np.int64),
#               np.random.randint(1,5, (64,1), dtype = np.int64), np.random.randint(1,5, (64,1), dtype = np.int64), np.random.rand(64,1)]
#     activations = ['relu', 'relu', 'relu', 'tanh', 'sigmoid', 'softmax', 'tanh']
#     messages = ['ignored because of the shape', 'Poor initialization (unbreaking symmetry) of weight', 'It is recommended to choose He initialization for weight',
#                 'It is recommended to choose Glorot/Xavier initialization for weight', 'It is recommended to choose Lecun initialization for weight',
#                 'It is recommended to choose a well-known initialization for weight', 'everything is clear']
#     for i in range(len(names)):
#         assert messages[i] in pre_check_weights(names[i], arrays[i], activations[i])


def pre_check_biases(initial_biases, inputs_data):
    app_path = Path.cwd()
    config_fpath = settings.load_user_config_if_exists(app_path)
    config = settings.Config(config_fpath).pre_check
    main_msgs = settings.load_messages()
    if not(initial_biases):
        return main_msgs['need_bias']
    else:
        checks = []
        for b_name, b_array in initial_biases.items():
            checks.append(np.sum(b_array)==0.0)
        if inputs_data.problem_type == CLASSIFICATION_KEY and \
            inputs_data.targets_metadata['balance'] < config.data.labels_perp_min_thresh:
            if checks[-1]:
                return main_msgs['last_bias']
            elif not checks[-1]:
                bias_indices = np.argsort(b_array)
                probas_indices = np.argsort(inputs_data.targets_metadata['probas'])
                if not (np.equal(bias_indices, probas_indices)).all():
                    return main_msgs['ineff_bias_cls']
        elif inputs_data.problem_type == REGRESSION_KEY:
            if inputs_data.targets_metadata['count'] == 1:
                avgs = [inputs_data.targets_metadata['mean']]
                stds = [inputs_data.targets_metadata['std']]
            else:
                avgs = list(inputs_data.targets_metadata['mean'])
                stds = list(inputs_data.targets_metadata['std'])
            var_coefs = [std/avg for avg, std in zip(avgs, stds)]
            low_var_coefs_indices = [i for i, var_coef in enumerate(var_coefs) if var_coef <= 1e-3]
            for idx in low_var_coefs_indices:
                b_value = float(b_array[idx])
                if not(utils.almost_equal(b_value, avgs[idx])):
                    return main_msgs['ineff_bias_regr'].format(idx)
        elif not np.all(checks):
            return main_msgs['zero_bias']

def test_biases():
    x_train0 = np.ones((314, 9), dtype=np.int64)
    y = np.array([[0],[1]])
    y_train0 = np.repeat(y, [1, 313], axis=0)
    x_test0 = np.ones((78, 9), dtype=np.int64)
    y_test0 = np.repeat(y, [1, 77], axis=0)
    data_loader_under_test = data.DataLoaderFromArrays(x_train0, y_train0, shuffle=True, one_hot=False, target_scaling=False)
    test_data_loader = data.DataLoaderFromArrays(x_test0, y_test0, shuffle=True, one_hot=False, target_scaling=False)
    data_to_test = interfaceData.build_data_interface(data_loader_under_test, test_data_loader, homogeneous=True)
    inputs_data0 = InputData(data_to_test, 'classification')
    initial_biases0 = {}
    assert pre_check_biases(initial_biases0, inputs_data0) == '''The model missed biases. Do not consider this alert if you use batchnorm for all layers'''
    initial_biases1 = {'dense/bias:0': np.zeros((64,1))}
    assert pre_check_biases(initial_biases1, inputs_data0) == '''Bias of last layer should not be zero, in case of unbalanced data'''
    initial_biases2 = {'conv2d:bias:0':np.ones((64,1))}
    assert pre_check_biases(initial_biases2, inputs_data0) == '''Bias of last layer should match the ratio of labels'''
    x_train1 = np.ones((314, 9), dtype=np.int64)
    x_test1 = np.ones((78, 9), dtype=np.int64)
    y_train1 = np.random.uniform(low = 143.50, high = 143.78, size = (314,1))
    y_test1 = np.random.uniform(low = 143.50, high = 143.78, size = (78,1))
    data_loader_under_test = data.DataLoaderFromArrays(x_train1, y_train1, shuffle=True, problem_type=REGRESSION_KEY, one_hot=False, target_scaling=False)
    test_data_loader = data.DataLoaderFromArrays(x_test1, y_test1, shuffle=True, one_hot=False, problem_type=REGRESSION_KEY, target_scaling=False)
    data_to_test = interfaceData.build_data_interface(data_loader_under_test, test_data_loader, homogeneous=True)
    inputs_data1 = InputData(data_to_test, 'regression')
    initial_biases3 = {'conv2d/bias:0': np.ones((64,1))}
    assert pre_check_biases(initial_biases3, inputs_data1) == '''Bias of last layer should start up with the mean value'''
    dataset = pd.read_csv('tests/data/auto-mpg.csv')
    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)
    train_features = train_dataset.copy()
    test_features = test_dataset.copy()
    train_labels = train_features.pop('MPG')
    test_labels = test_features.pop('MPG')
    x_train2, y_train2 = train_features.to_numpy(), train_labels.to_numpy().reshape(-1,1)
    x_test2, y_test2 = test_features.to_numpy(), test_labels.to_numpy().reshape(-1,1)
    data_loader_under_test = data.DataLoaderFromArrays(x_train2, y_train2, shuffle=True, problem_type=REGRESSION_KEY, one_hot=False, target_scaling=True)
    test_data_loader = data.DataLoaderFromArrays(x_test2, y_test2, shuffle=True, one_hot=False, problem_type=REGRESSION_KEY, target_scaling=True)
    data_to_test = interfaceData.build_data_interface(data_loader_under_test, test_data_loader, homogeneous=True)
    inputs_data2 = InputData(data_to_test, 'classification')
    initial_biases4 = {'conv2d/bias:0': np.ones((64,1))}
    assert pre_check_biases(initial_biases4, inputs_data2) == '''It is recommended to choose null biases (zeros)'''
 

def pre_check_loss(losses, inputs_data, initial_loss):
    app_path = Path.cwd()
    config_fpath = settings.load_user_config_if_exists(app_path)
    config = settings.Config(config_fpath).pre_check
    main_msgs = settings.load_messages()
    rounded_loss_rates = [round(losses[i + 1] / losses[i]) for i in range(len(losses) - 1)]
    equality_checks = sum(
        [(loss_rate == config.init_loss.size_growth_rate) for loss_rate in rounded_loss_rates])
    if equality_checks == len(rounded_loss_rates):
        return main_msgs['poor_reduction_loss']
    if inputs_data.problem_type == CLASSIFICATION_KEY:
        expected_loss = -np.log(1 / inputs_data.targets_metadata['labels'])
        err = np.abs(initial_loss - expected_loss)
        if err >= config.init_loss.dev_ratio * expected_loss:
            return main_msgs['poor_init_loss'].format(readable(err / expected_loss))

def test_losses():
    losses_list0 = [2.56, 4.5, 8.87]
    losses_list1 = [1, 1.5, 1.6]
    initial_loss0 = 1.54
    initial_loss1 = 2.01
    x_train0 = np.ones((314, 9), dtype=np.int64)
    y = np.array([[0],[1]])
    y_train0 = np.repeat(y, [1, 313], axis=0)
    x_test0 = np.ones((78, 9), dtype=np.int64)
    y_test0 = np.repeat(y, [1, 77], axis=0)
    data_loader_under_test = data.DataLoaderFromArrays(x_train0, y_train0, shuffle=True, one_hot=False, target_scaling=True)
    test_data_loader = data.DataLoaderFromArrays(x_test0, y_test0, shuffle=True, one_hot=False, target_scaling=True)
    data_to_test = interfaceData.build_data_interface(data_loader_under_test, test_data_loader, homogeneous=True)
    inputs_data = InputData(data_to_test, 'classification')
    assert pre_check_loss(losses_list0, inputs_data, initial_loss0) == 'The reduction of the loss is poorly designed: it is recommended to use AVG instead'
    assert 'Loss at cold start is considered poor and problematic:' in pre_check_loss(losses_list1, inputs_data, initial_loss1)

def pre_check_gradients(numerical, theoretical):
    app_path = Path.cwd()
    config_fpath = settings.load_user_config_if_exists(app_path)
    config = settings.Config(config_fpath).pre_check
    main_msgs = settings.load_messages()
    # def intermediate_function(x):
    #     return tf.convert_to_tensor(loss_value)
    # for i in range(len(weights)):
    #     theoretical, numerical = tf.test.compute_gradient(
    #         intermediate_function,
    #         x=[weights[i]],
    #         delta=config.grad.delta)
    #     print(theoretical)
    #
    weight_name = 'conv2d/bias:0'
    theoretical, numerical = theoretical[0].flatten(), numerical[0].flatten()
    total_dims, sample_dims = len(theoretical), int(config.grad.ratio_of_dimensions*len(theoretical))
    indices = np.random.choice(np.arange(total_dims), sample_dims, replace=False)
    theoretical_sample = theoretical[indices]
    numerical_sample = numerical[indices]
    numerator = np.linalg.norm(theoretical_sample - numerical_sample)
    denominator = np.linalg.norm(theoretical_sample) + np.linalg.norm(numerical_sample)
    if denominator == 0.0:
        return 'Unable to calculate the relative error between theoretical and numerical gradient: Gradients are equal to 0.\nThis checker will be ignored.'
    relerr = numerator / denominator
    if relerr > config.grad.relative_err_max_thresh:
        return main_msgs['grad_err'].format(weight_name, readable(relerr), config.grad.relative_err_max_thresh)

def test_gradients():
    numerical1 = [np.random.uniform(low=0.05, high=1.57, size=(64, 9))]
    theoretical1 = [np.random.uniform(low=0.05, high=1.57, size=(64, 9))]
    numerical0 = [np.zeros((64, 9))]
    theoretical0 = [np.zeros((64, 9))]
    assert pre_check_gradients(numerical0, theoretical0) == 'Unable to calculate the relative error between theoretical and numerical gradient: Gradients are equal to 0.\nThis checker will be ignored.'
    assert 'Gradient w.r.t weight' in pre_check_gradients(numerical1, theoretical1)


def pre_check_fitting_data_capability(real_loss, real_acc, fake_loss, problem_type):
        app_path = Path.cwd()
        config_fpath = settings.load_user_config_if_exists(app_path)
        config = settings.Config(config_fpath).pre_check
        main_msgs = settings.load_messages()
        def _loss_is_stable(loss_value):
            if np.isnan(loss_value).all():
                print(main_msgs['nan_loss'])
                return False, main_msgs['nan_loss']
            if np.isinf(loss_value).all():
                print(main_msgs['inf_loss'])
                return False, main_msgs['inf_loss']
            return (True,)
        if len(_loss_is_stable(real_loss)) == 2:
            return main_msgs['underfitting_single_batch']
        # loss, acc = (real_losses[-1] + self.model.reg_loss, real_accs[-1]) if self.model.reg_loss!= None else (real_losses[-1], real_accs[-1])
        underfitting_prob = False
        if problem_type == CLASSIFICATION_KEY:
            if 1.0 - max(real_acc) > config.prop_fit.mislabeled_rate_max_thresh:
                underfitting_prob = True
                return main_msgs['underfitting_single_batch']
        elif problem_type == REGRESSION_KEY:
            if min(real_acc) > config.prop_fit.mean_error_max_thresh:
                return main_msgs['underfitting_single_batch']
                underfitting_prob = True
        loss_smoothness = metrics.smoothness(real_loss)
        min_loss = np.min(real_loss)
        if min_loss <= config.prop_fit.abs_loss_min_thresh or (min_loss <= config.prop_fit.loss_min_thresh and loss_smoothness > config.prop_fit.smoothness_max_thresh):
            return main_msgs['zero_loss']
        # if not (underfitting_prob): return
            if not (_loss_is_stable(fake_loss)[0]): return
        stability_test = np.array([_loss_is_stable(loss_value)[0] for loss_value in (real_loss + fake_loss)])
        if (stability_test == False).any():
            last_real_losses = real_loss[-config.prop_fit.sample_size_of_losses:]
            last_fake_losses = fake_loss[-config.prop_fit.sample_size_of_losses:]
            if not (metrics.are_significantly_different(last_real_losses, last_fake_losses)):
                return main_msgs['data_dep']

def test_data_capability():
    real_loss1 = np.nan
    fake_loss1 = np.array([2.4])
    real_acc1 = [0.9]
    problem_type1 = ''
    assert pre_check_fitting_data_capability(real_loss1, real_acc1, fake_loss1, problem_type1) == 'The DNN training is unable to fit properly a single batch of data'

    real_loss2 = np.inf
    fake_loss2 = np.array([2.4])
    real_acc2 = [0.5]
    problem_type2 = ''
    assert pre_check_fitting_data_capability(real_loss2, real_acc2, fake_loss2, problem_type2) == 'The DNN training is unable to fit properly a single batch of data'

    real_loss3 = np.array([[1.548]])
    fake_loss3 = np.array([2.4])
    real_acc3 = [0.8]
    problem_type3 = 'classification'
    assert pre_check_fitting_data_capability(real_loss3, real_acc3, fake_loss3, problem_type3) == 'The DNN training is unable to fit properly a single batch of data'

    real_loss4 = np.array([[1.548]])
    fake_loss4 = np.array([2.4])
    real_acc4 = [0.1]
    problem_type4 = 'regression'
    assert pre_check_fitting_data_capability(real_loss4, real_acc4, fake_loss4, problem_type4) == 'The DNN training is unable to fit properly a single batch of data'

    real_loss5 = np.array([[1e-6]])
    fake_loss5 = np.array([2.4])
    real_acc5 = [0.0001]
    problem_type5 = 'regression'
    assert pre_check_fitting_data_capability(real_loss5, real_acc5, fake_loss5, problem_type5) == 'The loss is smoothly decreasing towards zero: The model may need regularization'

    real_loss6 = np.array([1.56, np.nan])
    fake_loss6 = np.array([2.4, np.nan])
    problem_type6 = ''
    real_acc6 = [0.95]
    assert pre_check_fitting_data_capability(real_loss6, real_acc6, fake_loss6, problem_type6) == 'The training procedure seems to be not considering the data inputs'
