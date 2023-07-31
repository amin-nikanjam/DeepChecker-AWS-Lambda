import os
import sys
from time import sleep
import json
import requests
import subprocess

def install_dependencies(repo_name):
    subprocess.run(['pip', 'install', '-r', f'{repo_name}/requirements.txt', '--target', '/tmp'])


import tensorflow as tf
from checkers import DeepChecker
import interfaceData as interfaceData
import data as data
from tensorflow.keras import datasets
  

def check_repo(owner, repo_name, token, pr_number):
    sys.path.append('/tmp')
    install_dependencies(repo_name)
    json_config = open(f'{repo_name}/config.json')
    json_config = json.load(json_config)
    filename=json_config["fileName"]
    sys.path.append(f'/tmp/{repo_name}')
    module=__import__(filename)
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    data_loader_under_test = data.DataLoaderFromArrays(x_train, y_train, shuffle=True, one_hot=True, normalization=True)
    test_data_loader = data.DataLoaderFromArrays(x_test, y_test, shuffle=True, one_hot=True, normalization=True)
    model = module.Model(x_train, y_train)
    data_under_test = interfaceData.build_data_interface(data_loader_under_test, test_data_loader, homogeneous=True)
    checker = DeepChecker(name='deep_checker_result', data=data_under_test, model=model, buffer_scale=10)
    checker.run_full_checks()
    write_output_as_github_comment(owner, repo_name, token, pr_number)


def write_output_as_github_comment(owner, repo_name, token, pr_number):
   log_file="deep_checker_result.log"
   with open(log_file, 'rb') as log_file:
    log_data = log_file.read().decode('utf-8')
   headers = {'Authorization': f'Bearer {token}', 'X-GitHub-Api-Version':'2022-11-28', 'Accept':'application/vnd.github+json'}
   comment_github_url=f'https://api.github.com/repos/{owner}/{repo_name}/issues/{pr_number}/comments'
   print('url: ', comment_github_url)
   data=json.dumps({"body":log_data})
   requests.post(comment_github_url, headers=headers, data=data)
