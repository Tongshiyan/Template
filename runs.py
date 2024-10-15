from experiment.EXP import *
if __name__ == '__main__':
    print('device : {}'.format(device))
    for ii in range(args.itr):
        base_args = args
        setting = '{}_{}_{}_lr{}_batch{}_opt{}_{}'.format(
            args.exp_name,
            args.task_name,
            args.model,
            args.learning_rate,
            args.batch_size,
            args.opt_name,
            ii + 1)
        exp = EXP_model(base_args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)
        torch.cuda.empty_cache()