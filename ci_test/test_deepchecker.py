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
from datetime import date


class DeepCheckerTest(unittest.TestCase):
    
    #this test checks that the niumber of bugs stated by deepchecker is as expected
    def test_deepchecker(self):
        expected_bug_occurences=62 
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
        
        
        #In order to get the number of bugs, we will need to get the date of today, because in each bug there is a timestapm where
        #the date is written in this format 2023-08-08 15.
        today = date.today()
        today = today.strftime("%Y-%m-%d")

        occurrences = count_occurrences(log_data, today)
        print('occ:', occurrences)
        print('occ2:', expected_bug_occurences)
        assert occurrences==expected_bug_occurences
        print('Test finished successfully.')


def count_occurrences(main_string, substring):
    count = 0
    start_index = 0

    while start_index < len(main_string):
        index = main_string.find(substring, start_index)
        if index == -1:
            break
        count += 1
        start_index = index + len(substring)
    return count



if __name__ == '__main__':
    unittest.main()