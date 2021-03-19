# ############################################
# Model learning of two link robot arm
# Script for training and testing of model
#
# Author : Deepak Raina @ IIT Delhi
# Version : 0.1
# ############################################

from controller import RobotController

# Robot controller
controller = RobotController()
EPOCHS = 10000
MODEL_FILE_LOC = 'models/trained_nn_model_' + str(EPOCHS)

## Training
controller.train(epochs=EPOCHS)

## Testing
# controller.test(model_fileloc = MODEL_FILE_LOC, num_test=1)
