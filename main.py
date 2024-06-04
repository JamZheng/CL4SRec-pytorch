import torch
import os
import argparse
import logging
import sys
from datetime import datetime
from time import time

sys.path.append('..')
from tensorboardX import SummaryWriter
from trainer import Trainer
from model import CL4SRec
from data_iterator import TrainData, TestData
from tool import trans_to_cuda, setup_seed

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='Beauty')
    parser.add_argument('--model', type=str, default='CL4SRec')
    # contrastive learning 
    parser.add_argument('--cl_embs', type=str, default='concat', help='concat | predict | mean')
    parser.add_argument('--w_clloss', type=float, default=0.1, help='the weight of cl loss')
    parser.add_argument('--temp', type=float, default=1)
    parser.add_argument('--aug_type', type=str, default= 'mcr',
                        help='mask | crop | reorder')
    parser.add_argument('--is_hard', type=bool, default=True)
    parser.add_argument('--mask_p', type=float, default=0.5)
    parser.add_argument('--crop_p', type=float, default=0.6)
    parser.add_argument('--reorder_p', type=float, default=0.2)


    parser.add_argument('--repeat_rec', type=int, default=0)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout_prob', type=float, default=0.25)
    parser.add_argument('--filename', type=str, default='debug', help='post filename')
    parser.add_argument('--random_seed', type=int, default=11)
    parser.add_argument('--embedding_size', type=int, default=64)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--inner_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--hidden_dropout_prob', type=float, default=0.5)
    parser.add_argument('--attention_probs_dropout_prob', type=float, default=0.5)
    parser.add_argument('--num_hidden_layers', type=int, default=2)
    parser.add_argument('--num_attention_heads', type=int, default=2)
    parser.add_argument('--hidden_act', type=str, default='gelu')
    parser.add_argument('--lr', type=float, default=0.001, help='')
    parser.add_argument('--weight_decay', type=float, default=0.000, help='')
    parser.add_argument('--max_iter', type=int, default=100, help='(k)')
    parser.add_argument('--maxlen', type=int, default=50)
    parser.add_argument('--best_ckpt_path', type=str, default='runs/',
                        help='the direction to save ckpt')
    parser.add_argument('--log_dir', type=str, default='log_debug', help='the direction of log')
    parser.add_argument('--loss_type', type=str, default='BCE', help='CE | BCE')

    parser.add_argument('--test_epoch', type=int, default=1)
    parser.add_argument('--patience', type=int, default=40)
    parser.add_argument('--max_epoch', type=int, default=500)

    return parser.parse_args()



def main():
    # initial config and seed
    config = get_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu_id)
    os.environ['CUDA_LAUNCH_BLOCKING']='1'
    SEED = config.random_seed
    setup_seed(SEED)

    if not os.path.exists(config.log_dir):
        os.mkdir(config.log_dir)
    config.log_dir += '/' + datetime.now().strftime('%m%d')
    if not os.path.exists(config.log_dir):
        os.mkdir(config.log_dir)
    # config.log_dir = os.path.join(config.log_dir, datetime.now().strftime('%m%d'))

    filename = '{}_{}_{}_{}_{}_{}'.format(config.dataset, config.model,config.batch_size, config.aug_type, config.cl_embs,
                                                            datetime.fromtimestamp(time()).strftime('%m%d%H%M'))

    config.best_ckpt_path += filename
    if not os.path.exists('runs_tensorboard'): os.mkdir('runs_tensorboard')
    if not os.path.exists('runs'): os.mkdir('runs')

    logging.basicConfig(format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
                        filename='{}/{}.log'.format(config.log_dir, filename),
                        level=logging.INFO)
    logger = logging.getLogger(__name__)
    config.logger = logger

    # initial dataset
    train_data = TrainData(config)
    valid_data = TestData(config, is_valid=True)
    test_data = TestData(config, is_valid=False)
    config.n_item, config.n_user = train_data.num_items + 1, train_data.num_users + 1

    writer = SummaryWriter('runs_tensorboard/{}'.format(filename))

    logger.info("--------------------Configure Info:------------")
    for arg in vars(config):
        logger.info(f"{arg} : {getattr(config, arg)}")

    # initial model
    model = trans_to_cuda(CL4SRec(config))
    
    # initial trainer
    trainer = Trainer()

# ------------------train------------------------------
    best_metrics = [0]
    trials = 0
    best_epoch = 0

    for i in range(config.max_epoch):
        epoch = i + 1
        trainer.train(epoch, writer, model, train_data, config)
        
        if epoch % config.test_epoch == 0:
            metrics = trainer.eval(epoch, model, config, valid_data, [5, 20], phase='valid')
            
            
            if metrics[-1] > best_metrics[-1]:
                best_epoch = epoch
                torch.save(model.state_dict(), config.best_ckpt_path)
                best_metrics = metrics
                trials = 0
            else:
                trials += 1
                # early stopping
                if trials > config.patience:
                    break

# ------------------test------------------------------
    model.load_state_dict(torch.load(config.best_ckpt_path))
    logger.info('-------------best valid in epoch {:d}-------------'.format(best_epoch))
    trainer.eval(epoch, model, config, valid_data, ks = [5, 20], phase='valid')
    logger.info('------------test-----------------')
    trainer.eval(epoch, model, config, test_data, ks=[5, 20], phase='test')

if __name__ == "__main__":
    main()
