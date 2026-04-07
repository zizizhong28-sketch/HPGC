import os
import hydra
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from models import *
from HEM.dataloaders.hem_dataloader import HEMLoader
from pytorch_lightning.loggers import WandbLogger
import torch

gpu_count = torch.cuda.device_count()
for i in range(gpu_count):
    gpu_name = torch.cuda.get_device_name(i)
    print(f"GPU {i}: {gpu_name}")
os.environ["WANDB_MODE"] = "offline"
# os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

@hydra.main(version_base=None, config_path="configs", config_name="train_obj.yaml")
def main(cfg):

    # # random seed
    seed_everything(cfg.train.seed, workers=True)

    # set model
    model_name = cfg.model.class_name
    model_class = eval(model_name)

    if cfg.train.load_pretrain:
        print('Loading pretrained weights from', cfg.train.load_ckpt)
        model = model_class.load_from_checkpoint(cfg.train.load_ckpt, map_location=torch.device('cuda:0'), strict=True)
        # missing_keys, unexpected_keys = model.load_pretrain(cfg.train.load_pretrain, strict=True)
        # print(missing_keys)
        # print(unexpected_keys)
    else:
        print('Training from blank weights')
        model = model_class(cfg)

    # set data
    dataloader = HEMLoader(cfg.data)

    # train
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    save_path = hydra_cfg['runtime']['output_dir'] + '/ckpt'
    print('saving in', save_path[save_path.find('outputs'):-5])

    trainer = Trainer(max_epochs=cfg.train.epoch,
                    default_root_dir=save_path,
                    deterministic=False,
                    accelerator='gpu',
                    devices=cfg.gpus,
                    logger=WandbLogger(project='wandb_output'),
                    callbacks=[
                        ModelCheckpoint(save_path, save_top_k=-1),
                        EarlyStopping(monitor="occupancy_loss", mode="min")
                    ],
    )
    
    trainer.fit(model, dataloader)


if __name__ == '__main__':
    main()
