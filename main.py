
import hydra
from omegaconf import DictConfig, OmegaConf

from dataset import prepare_dataset
from client import generate_client_fn

@hydra.main(config_path="conf", config_name="base", version_base=None)

def main(cfg: DictConfig):
    #1 parse config & get experiment output dir
    print(OmegaConf.to_yaml(cfg))
    #2 prepare dataset
    trainloaders, validationloaders, testloader = prepare_dataset(cfg.num_clients,
                                                                cfg.batch_size)


    #3 define clients 
    client_fn = generate_client_fn(trainloaders, validationloaders, cfg.num_classes)

    

if __name__ == "__main__":
    main()