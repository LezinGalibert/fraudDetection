# fraudDetection
Fraud detection exercise for the Ubisoft application.

System requirements:
    * scikit-learn==0.22.1
    * pandas==0.23.4
    * numpy==1.15.1
    * Flask==1.0.2
    * joblib==0.14.1
    
This repository contains classes and scripts used to solve the problem stated in Mle_fraud_usecase.pdf .
Scripts include:
    ** Scripts
        * fraudDetectionClass.py: class to manage labelled and unlabelled data from the dataset, and prepare the data for the model training.
        * assembleAdaBoostClass.py: class for building the based model presented in http://homepages.rpi.edu/~bennek/kdd-KristinBennett1.pdf . Also contains the function that answers question 3.
        * prediction.py: main data pipeline which loads the data, initialize the model and implements a prediction function that accepts a json format data.
        * server.py: creates the flask app that allows the user to call the pipeline and show the results of a prediction.
        * start_flask.sh: launches the flask app.
    ** Docker
        * Dockerfile: docker image.
        * docker-compose.yml: creates the app container.
        * python-requirements.txt: package requirements for the image.
    ** Other:
        * __init__py: python helper for loading classes.
        * toPredict.json: example json file for prediction.
        
Question summary:
1) The assembleAdaBoost class tried as much as possible to match the pseudo code detailled in the paper. However, some adjustment are most probably to be made as it was not possible at this stage to maintain Ft(xi) between 0 and 1 (a probablity). A sigmoid function can be applied to Ft(xi) to place it be 0 and 1, but this step is not mentionned in the pseudo code.

2) The training pipeline can be found in prediction.py and a confusion matrix is printed out when the flask app is called, for the training and the test sets.

3) The optimal decision for a transaction corresponds to the expectation for this decision to be greater than the expectation for the other decision, eg
E(order is blocked) > E(order is not blocked) means that we will block the order.

For each event, we can write E(decision) = p(fraud) * amount(fraud | decision) + (1 - p) * fraud * amount(not fraud | decision), with amount(fraud/not fraud | decision) being the amount of money we make (or lose) when we take the decision and there is / is not fraud.
If we block the order, no money is made or lost, so E(order is blocked) = 0.
If the order is not blocked, amount(fraud) = -F and amount(not fraud) = M, and therefore
E(order is not blocked) = - p * F + (1 - p) * M.
The good decision is then when E(order is blocked) > E(order is not blocked) ie:
                                        0          > - p * F + (1 - p) * M

We can see this makes sense as:
* if F is very high, the penalty to not block the order will be very high. The term - p * F dominates, unless p is very small which means that we need to be almost 100% that the order is not fraudulent to not block it.
* if M is very high, the potential gain is huge. The term (1 - p) * M dominates, unless p is very close to 1, which means that we need to be almost 100% sure that the order is fraudulent to block it.

4) The Docker folder contains all elements to launch the flask app in your browser.
Please use the following command to see the output prediction:
-->  curl -X POST -H "Content-Type: application/json" -d @toPredict.json http://localhost:5000/score  <--
