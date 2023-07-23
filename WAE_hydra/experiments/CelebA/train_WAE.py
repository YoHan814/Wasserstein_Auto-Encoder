import os, sys, logging, importlib
CODE_HOME = os.path.normpath(os.path.join(os.getcwd(), "..", ".."))
sys.path.append(CODE_HOME)
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

import hydra
#from hydra import compose, initialize
from hydra.utils import instantiate, get_original_cwd
from omegaconf import OmegaConf, open_dict

log = logging.getLogger(__name__)
LOG_NAME = "wae"
OmegaConf.register_new_resolver("lname", lambda : LOG_NAME)

@hydra.main(config_path='../../configs/', config_name='train.yaml')
def train(cfg) -> None:
    # with open_dict(cfg):
    #     cfg.train_info.path_ckpt = os.path.join(get_original_cwd(), cfg["train_info"]["checkpoint_path"])
    dataset = instantiate(cfg['train_info']['data'])
    mod = importlib.import_module('src.model.' + cfg['train_info']['model'])
    mod_attr = getattr(mod, cfg['train_info']['architecture'])
    network = mod_attr(cfg, log)
    network.log_architecture()
    logger = TensorBoardLogger(".", "", "", log_graph = True, default_hp_metric=False)
    trainer = Trainer(gpus = 1, strategy = "dp", max_epochs = cfg['train_info']['epoch'], check_val_every_n_epoch=1, logger = logger, callbacks = RichProgressBar(5))
    trainer.fit(network, dataset)
    if cfg["path_info"]["manual_save"] is not None:
        trainer.save_checkpoint(cfg["path_info"]["manual_save"])

if __name__ == "__main__":
    train()