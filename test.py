import os
import math
import argparse
import random
import logging
import torch
import numpy as np
from tqdm import tqdm
import options.base_options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model
from collections import OrderedDict


def main():
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, required=True, default=None, help='Path to option YAML file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=False)

    #### distributed training settings

    opt['dist'] = False
    rank = 1
    print('Disabled distributed training.')

    #### loading resume state if exists
    if opt['path'].get('resume_state', None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'],
                                  map_location=lambda storage, loc: storage.cuda(device_id))
        option.check_resume(opt, resume_state['iter'])  # check resume options
    else:
        resume_state = None

    #### mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0)
        if resume_state is None:
            util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                         and 'pretrain_model' not in key and 'resume' not in key))

        # config loggers. Before it, the log will not work
        util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
    else:
        util.setup_logger('base', opt['path']['root'], 'test_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)

    logger = logging.getLogger('base')

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    #### random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    logger.info('Random seed: {}'.format(seed))
    util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True

    #### create test dataloader
    # dataset_ratio = 200  # enlarge the size of each epoch
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'test':
            test_set = create_dataset(opt, dataset_opt)
            test_loader = create_dataloader(test_set, dataset_opt, opt, None)
            logger.info('Number of test images in [{:s}]: {:d}'.format(
                dataset_opt['name'], len(test_set)))

    #### create model
    model = create_model(opt)

    #### resume training
    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))

        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    img_dir = opt['path']['test_images']
    util.mkdir(img_dir)

    if opt['datasets'].get('test', None):
        avg_psnr_exp1 = []
        avg_psnr_exp2 = []
        avg_psnr_exp3 = []
        avg_psnr_exp4 = []
        avg_psnr_exp5 = []
        avg_ssim_exp1 = []
        avg_ssim_exp2 = []
        avg_ssim_exp3 = []
        avg_ssim_exp4 = []
        avg_ssim_exp5 = []
        for test_data in tqdm(test_loader):

            model.feed_data(test_data)
            model.test()

            out_dict = OrderedDict()
            out_dict['LQs'] = model.var_L.detach().float().cpu()
            out_dict['rlts'] = model.fake_H.detach().float().cpu()
            out_dict['GTs'] = model.real_H.detach().float().cpu()

            # Save SR images for reference
            for i in range(len(test_data['LQ_path'])):
                img_path = test_data['LQ_path'][i]
                # get pure name without .jpg and so on
                img_name = '.'.join(os.path.basename(img_path).split(".")[:-1])

                gt_img = util.tensor2img(out_dict['GTs'][i])
                en_img = util.tensor2img(out_dict['rlts'][i])

                # calculate metrics
                psnr_inst = util.calculate_psnr(en_img, gt_img)
                ssim_inst = util.calculate_ssim(en_img, gt_img)
                if math.isinf(psnr_inst) or math.isnan(psnr_inst) or \
                        math.isinf(ssim_inst) or math.isnan(ssim_inst):
                    psnr_inst = 0
                    ssim_inst = 0

                suffix = img_name.split('_')[-1]
                if suffix == '0':
                    avg_psnr_exp1.append(psnr_inst)
                    avg_ssim_exp1.append(ssim_inst)
                elif suffix == 'N1':
                    avg_psnr_exp2.append(psnr_inst)
                    avg_ssim_exp2.append(ssim_inst)
                elif suffix == 'N1.5':
                    avg_psnr_exp3.append(psnr_inst)
                    avg_ssim_exp3.append(ssim_inst)
                elif suffix == 'P1':
                    avg_psnr_exp4.append(psnr_inst)
                    avg_ssim_exp4.append(ssim_inst)
                elif suffix == 'P1.5':
                    avg_psnr_exp5.append(psnr_inst)
                    avg_ssim_exp5.append(ssim_inst)
                else:
                    raise FileNotFoundError("File is not found.......")

        avg_psnr_all = sum(avg_psnr_exp1) + sum(avg_psnr_exp2) + sum(avg_psnr_exp3) \
                       + sum(avg_psnr_exp4) + sum(avg_psnr_exp5)
        avg_ssim_all = sum(avg_ssim_exp1) + sum(avg_ssim_exp2) + sum(avg_ssim_exp3) \
                       + sum(avg_ssim_exp4) + sum(avg_ssim_exp5)
        count = len(avg_psnr_exp1) + len(avg_psnr_exp2) + len(avg_psnr_exp3) \
                + len(avg_psnr_exp4) + len(avg_psnr_exp5)

        logger.info('# Test # Average PSNR: {:.4f}, Average SSIM: {:.4f}.'.
                    format(avg_psnr_all / count, avg_ssim_all / count))
        logger.info('# Test # PSNR 0: {:.4f}, PSNR N1: {:.4f}, PSNR N1.5: {:.4f}, PSNR P1: {:.4f}, PSNR P1.5: {:.4f}.'.
                    format(np.mean(avg_psnr_exp1), np.mean(avg_psnr_exp2),
                           np.mean(avg_psnr_exp3), np.mean(avg_psnr_exp4), np.mean(avg_psnr_exp5)))
        logger.info('# Test # SSIM 0: {:.4f}, SSIM N1: {:.4f}, SSIM N1.5: {:.4f}, SSIM P1: {:.4f}, SSIM P1.5: {:.4f}.'.
                    format(np.mean(avg_ssim_exp1), np.mean(avg_ssim_exp2),
                           np.mean(avg_ssim_exp3), np.mean(avg_ssim_exp4), np.mean(avg_ssim_exp5)))

    logger.info('End of training.')


if __name__ == '__main__':
    main()
