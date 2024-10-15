from experiment.EXP import *
if __name__ == '__main__':

    for ii in range(args.itr):
        for args.condition_mode in range(3):
            base_args=args
            setting = '{}_{}_Condition{}_Patch{}_embed{}_block{}_h{}_w{}_lr{}_batch{}_opt{}_{}'.format(
                args.exp_name,
                args.model,
                3-args.condition_mode,
                args.patch_size,
                args.embed_dim,
                args.block_num,
                args.h,
                args.w,
                args.learning_rate,
                args.batch_size,
                args.opt_name,
                ii+1)
            exp = EXP_model(base_args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.vail_img_plot(setting)
            torch.cuda.empty_cache()