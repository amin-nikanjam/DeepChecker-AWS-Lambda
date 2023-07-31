import unittest
import os
os.chdir("ci_test")
import sys
sys.path.append('..')
import tensorflow as tf
from checkers import DeepChecker
import interfaceData as interfaceData
import data as data
from tensorflow.keras import datasets
import CNN_with_high_lr as module

class DeepCheckerTest(unittest.TestCase):
    
    def test_deepchecker(self):
        expected_data=""""""
        print('Test staring..')
        (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
        data_loader_under_test = data.DataLoaderFromArrays(x_train, y_train, shuffle=True, one_hot=True, normalization=True)
        test_data_loader = data.DataLoaderFromArrays(x_test, y_test, shuffle=True, one_hot=True, normalization=True)
        model = module.Model(x_train, y_train)
        data_under_test = interfaceData.build_data_interface(data_loader_under_test, test_data_loader, homogeneous=True)
        checker = DeepChecker(name='deep_checker_result', data=data_under_test, model=model, buffer_scale=10)
        checker.run_full_checks()
        log_file="deep_checker_result.log"
        with open(log_file, 'rb') as log_file:
            log_data = log_file.read().decode('utf-8')
        print('log data file:', log_data)
        print('expected data:', expected_data)
        # Find the minimum length of the two strings
        min_length = min(len(log_data), len(expected_data))
        
        # Iterate through the characters of both strings up to the minimum length
        for i in range(min_length):
            if log_data[i] != expected_data[i]:
                print(f"Differing character at position {i}: '{log_data[i]}' and '{expected_data[i]}'")
        
        # If the strings have different lengths, print the extra characters in the longer string
        if len(log_data) > min_length:
            for i in range(min_length, len(log_data)):
                print(f"Extra character in the first string at position {i}: '{log_data[i]}'")
        elif len(expected_data) > min_length:
            for i in range(min_length, len(expected_data)):
                print(f"Extra character in the second string at position {i}: '{expected_data[i]}'")


        assert log_data==expected_data
        print('Test finished successfully.')




if __name__ == '__main__':
    unittest.main()