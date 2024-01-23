import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
import torch
import argparse
import warnings
import pytorch_lightning as pl
from pytorch_lightning import Trainer, strategies
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import CSVLogger
from data_provider.stage2_dm import Stage2DM
from model.unimol import SimpleUniMolModel
from model.blip2_stage2 import Blip2Stage2
from model.dist_funs import MyDeepSpeedStrategy
from model.llama_flash_attention import replace_llama_attn_with_flash_attn
import pickle

os.environ['OPENBLAS_NUM_THREADS'] = '1'
## for pyg bug
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
## for A100 gpus
torch.set_float32_matmul_precision('medium')


def main(args):
    pl.seed_everything(args.seed)
    # model
    if args.init_checkpoint:
        model = Blip2Stage2.load_from_checkpoint(args.init_checkpoint, strict=False, args=args)
        print(f"loaded init checkpoint from {args.init_checkpoint}")
    elif args.stage2_path:
        model = Blip2Stage2(args)
        ckpt = torch.load(args.stage2_path, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'], strict=False)
        print(f"loaded stage2 model from {args.stage2_path}")
    elif args.stage1_path:
        model = Blip2Stage2(args)
        model.load_from_stage1_checkpoint(args.stage1_path)
        print(f"loaded stage1 model from {args.stage1_path}")
    else:
        model = Blip2Stage2(args)

    print(' total params:', sum(p.numel() for p in model.parameters()))

    tokenizer = model.blip2opt.llm_tokenizer
    # data

    dm = Stage2DM(args.mode, args.num_workers, args.batch_size, args.root, args.text_max_len, model.blip2opt.dictionary, tokenizer, args)

    callbacks = []
    callbacks.append(plc.ModelCheckpoint(dirpath="all_checkpoints/"+args.filename+"/", 
                                         filename='{epoch:02d}', 
                                         every_n_epochs=args.save_every_n_epochs, 
                                         save_last=True, 
                                         save_top_k=-1,
                                         save_on_train_epoch_end=True))
    if len(args.devices.split(',')) > 1:
        if args.strategy_name == 'deepspeed':
            strategy = MyDeepSpeedStrategy(stage=2)
        else:
            strategy = strategies.DDPStrategy(start_method='spawn')
    else:
        strategy = 'auto'
        args.devices = eval(args.devices)
    logger = CSVLogger(save_dir=f'./all_checkpoints/{args.filename}/')

    trainer = Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        max_epochs=args.max_epochs,
        accumulate_grad_batches=args.accumulate_grad_batches,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        callbacks=callbacks,
        strategy=strategy,
        logger=logger,
    )

    if args.mode == 'pretrain':
        trainer.fit(model, datamodule=dm)
    elif args.mode == 'ft':
        trainer.fit(model, datamodule=dm)
        trainer.test(model, datamodule=dm)
    elif args.mode == 'eval':
        trainer.fit_loop.epoch_progress.current.completed = args.caption_eval_epoch - 1
        trainer.test(model, datamodule=dm)
    else:
        raise NotImplementedError()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default="stage2")
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    # MM settings
    parser.add_argument('--mode', type=str, default='eval')
    parser.add_argument('--strategy_name', type=str, default='deepspeed')
    parser.add_argument('--use_3d', action='store_true', default=True)
    parser.add_argument('--enriched_descrption', action='store_true', default=False)
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', type=str, default='0,1,2,3')
    parser.add_argument('--precision', type=str, default='bf16-mixed')
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--accumulate_grad_batches', type=int, default=2)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=2)
    parser.add_argument('--enable_flash', action='store_true', default=True)
    parser = Blip2Stage2.add_model_specific_args(parser)
    parser = Stage2DM.add_model_specific_args(parser)
    parser = SimpleUniMolModel.add_args(parser)
    args = parser.parse_args()

    if args.enable_flash:
        replace_llama_attn_with_flash_attn()
    print("=========================================")
    for k, v in sorted(vars(args).items()):
        print(k, '=', v)
    print("=========================================")
    return args


if __name__ == '__main__':
    main(get_args())

