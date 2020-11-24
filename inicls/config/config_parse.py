import argparse
def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-net', '--network', dest='network', type=str, default=False,
                        help='choose a network')
    parser.add_argument('-opt', '--optimizer', dest='optimizer', type=str, default=False,
                        help='choose an optimizer')
    parser.add_argument('-sche', '--scheduler', dest='scheduler', type=str, default=False,
                        help='choose a scheduler')
    parser.add_argument('-loss', '--loss', dest='loss', type=str, default=False,
                        help='choose a loss')
    parser.add_argument('-lr', '--learning-rate', metavar='LR', type=float, nargs='?', default=1,
                        help='Learning rate', dest='lr')
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=1,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=16,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-n', '--name', dest='name', type=str, default="",
                        help='train name')

    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-train', '--train_csv', dest='train_csv', type=str, default=False,
                        help='train csv file_path')
    parser.add_argument('-valid', '--valid_csv', dest='valid_csv', type=str, default=False,
                        help='valid csv file_path')
    parser.add_argument('-test', '--test_csv', dest='test_csv', type=str, default=False,
                        help='test csv file_path')

    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')

    return parser.parse_args()