To use the applications, you need to do the following:
1.	Create a GitHub app (you can follow this link: https://docs.github.com/en/apps/creating-github-apps/registering-a-github-app/registering-a-github-app)
The role of the GitHub app is to listen to Pull Requests and send a notification to AWS if there is one.
For the “webhook URL” field we will put the URL of AWS API Gateway.
2.	Create the API Gateway, create a POST request and link it to AWS Lambda for execution.
3.	Create a Lambda function.
Now our listeners are ready, so we need to put the handling function inside AWS Lambda.
4.	Clone (exp: Neuralint or TheDeepChecker) from Github repo.
5.	Build the docker image
6.	Tag it
7.	Create AWS ECR repository
8.	Push the image to ECR
9.	Upload the docker image from ECR to AWS Lamba (Lambda container image is a new feature of AWS).
Now everything is put into place, to test it we do the following:
10.	Push a test repo in GitHub as an independent repository.
11.	Install the GitHub app inside the test repo 
12.	Create a branch
13.	Do a pull request
14.	Expect the tool (e.g, Neuralint) report as a comment.
