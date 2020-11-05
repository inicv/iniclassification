import os

# img_path
dir_raw_train = ""
dir_raw_test = ""

# submission_path
dir_submission = ""

# weight_path
dir_weight = ''

# csv_path
dir_csv_train = ''
dir_csv_test = os.path.join(dir_submission, 'test.csv')



seed_random = 2020

num_classes = 6
num_epochs = 100
num_patience_epoch = 10
num_KFold = 5

size_valid = 0.1
step_train_print = 200

factor_train = 1.25
size_train_image = 256
size_valid_image = 256
size_test_image = 256

batch_size = 64


predict_mode = 1

model_name = 'resnet18'
save_model_name = 'resnet18'
predict_model_names = "resnet18"
