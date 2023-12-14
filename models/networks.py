import models.archs.EnhanceN_arch as EnhanceN_arch6


# Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    if which_model == 'RDPNet':
        netG = EnhanceN_arch6.InteractNet(nc=opt_net['nc'])
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG
