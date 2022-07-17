import argparse

def argument_parser():
    parser = argparse.ArgumentParser(description="attribute recognition",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--dataset", type=str, default="PA100K")
    parser.add_argument("--num-att", type=int,default=26)
    parser.add_argument("--debug", action='store_false')

    parser.add_argument("--batchsize","-b", type=int, default=32)
    parser.add_argument("--train-epoch", type=int, default=25)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--lr-ft", type=float, default=0.01, help='learning rate of feature extractor')
    parser.add_argument("--lr-new", type=float, default=0.1, help='learning rate of classifier_base')
    parser.add_argument('--classifier', type=str, default='base', help='classifier name')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=0.0005)
    parser.add_argument("--train_split", type=str, default="trainval", choices=['train', 'trainval'])
    parser.add_argument("--valid_split", type=str, default="test", choices=['test', 'valid'])
    parser.add_argument("--redirector", action='store_false')
    parser.add_argument('--use_bn', action='store_false')

    parser.add_argument('--ratio', default=1,type=float, help='')
    parser.add_argument('--pos-weight', default=1,type=float, help='')
    parser.add_argument('--qratio', default=1,type=float, help='')
    parser.add_argument('--lratio', default=1,type=float, help='')
    parser.add_argument('--ldropout', default=0.3,type=float, help='')
    parser.add_argument('--after-norm', default=False, action='store_true')

    parser.add_argument('--gpu', default=6, type=int, help='GPU id to use.')
    parser.add_argument('--seed',default=0, type=int)
    
    parser.add_argument('--randAug', default=False,action='store_true',help='random augmentation')
    parser.add_argument('--n', default=2,type=int,help='random augmentation')
    parser.add_argument('--m', default=9,type=int,help='random augmentation')
    parser.add_argument('--labeldp',default=0.1, type=float)
    parser.add_argument('--beta', default=0, type=float, help='GPU id to use.')
    parser.add_argument('--mp', default=0.5, type=float, help='GPU id to use.')

    parser.add_argument('--hidden-dim', default=2048, type=int)
    parser.add_argument('--dim-forward', default=2048, type=int)
    parser.add_argument('--decN', default=2, type=int)
    parser.add_argument('--decNL', default=2, type=int)
    parser.add_argument('--dropout', default=0.1, type=float,help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=4, type=int)
    return parser
