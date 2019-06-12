#_*_coding:utf-8_*_

#author: lgz

#date: 19-6-10


import argparse

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):

        self.initialized = True

    def parse(self):

        if not self.initialized:
            self.initialize()

        self.opt = self.parser.parse_args()

        args = vars(self.opt)

        print('======================Check options========================')
        print('')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('')
        print('======================check done============================')


        return self.opt

class TrainOptions(Options):
    def initialize(self):
        Options.initialize(self)
        self.parser.add_argument("--num_epoch", action="store", type=int,
                                 default= 10 ,
                                 help="training epoches")

        self.parser.add_argument("--finetune_last_layer", action="store", type=bool,
                                 default=False,
                                 help="if True, just finetune the last layers")

        self.parser.add_argument("--resize", action="store", type=int,
                                default=224,
                                help="resize input")

        self.parser.add_argument("--dataset_dir", action="store", type=str,
                                default="data/hymenoptera_data",
                                help="cifar10, hymenoptera_data, where the dataset put in ")

        self.parser.add_argument("--save_model", action="store", type=str,
                                default="save_model",
                                help="where the model save ")

        self.parser.add_argument("--batchsize", action="store", type=int,
                                 default=16,
                                 help="training batchsize")

        self.parser.add_argument('--loss_type', type=str, default='CosFace',
                                 help='ArcFace, CosFace, Softmax')


        self.parser.add_argument('--backbone', type=str, default='resnet18',
                                 help='resnet18, densenet169, ')


        self.parser.add_argument('--save_loss_dir', type=str, default='show_loss_value',
                                 help='record the loss value ')






