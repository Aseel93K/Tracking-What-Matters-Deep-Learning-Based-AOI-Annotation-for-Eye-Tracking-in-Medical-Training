import torch
import mmcv
import mmengine
import mmdet
from mmengine import Config
from mmengine.runner import set_random_seed
import yaml
from mmengine.runner import Runner
import ast

yaml_file_path = "config.yml"

with open(yaml_file_path, 'r') as file:
    data = yaml.safe_load(file)

print(data)


def print_environment_info():
    print("mmengine:", mmengine.__version__)
    print("mmcv:", mmcv.__version__)
    print("mmdetection:", mmdet.__version__)
    print("torch version:", torch.__version__, "cuda:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA Device:", torch.cuda.current_device())
        print(torch.cuda.get_device_name(0))


def setup_cfg(config_path, classes, load_from, work_dir):
    cfg = Config.fromfile(config_path)

    # Set class names
    cfg.metainfo = {
        'classes': classes
    }

    # Dataset paths
    cfg.train_dataloader.batch_size = 50
    cfg.val_dataloader.batch_size = 50
    cfg.test_dataloader.batch_size = 50

    cfg.data_root = './dataset'
    cfg.train_dataloader.dataset.ann_file = 'valid/_annotations_filtered.coco.json'
    cfg.train_dataloader.dataset.data_root = cfg.data_root
    cfg.train_dataloader.dataset.data_prefix.img = 'valid/'
    cfg.train_dataloader.dataset.metainfo = cfg.metainfo

    cfg.val_dataloader.dataset.ann_file = 'test/_annotations_filtered.coco.json'
    cfg.val_dataloader.dataset.data_root = cfg.data_root
    cfg.val_dataloader.dataset.data_prefix.img = 'test/'
    cfg.val_dataloader.dataset.metainfo = cfg.metainfo

    cfg.train_cfg.max_epochs = 1
    #test
    cfg.test_dataloader.dataset.data_root = cfg.data_root
    cfg.test_dataloader.dataset.ann_file = 'test/_annotations_filtered.coco.json'
    cfg.test_dataloader.dataset.data_prefix.img = 'test/'
    cfg.test_dataloader.dataset.metainfo = cfg.metainfo

    # Evaluator
    cfg.val_evaluator.ann_file = cfg.data_root + '/' + 'valid/_annotations_filtered.coco.json'
    cfg.test_evaluator = cfg.val_evaluator

    # Number of classes
    cfg.model.roi_head.bbox_head.num_classes = 2

    # Pretrained weights
    cfg.load_from = load_from

    # Output and checkpoints
    cfg.work_dir = work_dir
    cfg.train_cfg.val_interval = 1
    cfg.default_hooks.checkpoint.interval = 1

    # Adjust learning rate for single GPU
    cfg.optim_wrapper.optimizer.lr = 0.005
    cfg.default_hooks.logger.interval = 10
    cfg.custom_imports = dict(
        imports=['custom_hooks.val_loss'],
        allow_failed_imports=False
    )
    cfg.custom_hooks = [
        dict(type='ValLoss')  # You can also pass additional args here
    ]

    # Set seed for reproducibility
    set_random_seed(0, deterministic=False)

    # Default hooks
    cfg.default_hooks = dict(
        timer=dict(type='IterTimerHook'),
        logger=dict(type='LoggerHook', interval=1),
        param_scheduler=dict(type='ParamSchedulerHook'),
        checkpoint=dict(type='CheckpointHook', interval=1),
        sampler_seed=dict(type='DistSamplerSeedHook'),
        visualization=dict(type='DetVisualizationHook')
    )

    # TensorBoard logging
    cfg.visualizer.vis_backends.append({"type": 'TensorboardVisBackend'})

    # new
    # Enable logging to file
    cfg.env_cfg.log_file = f'{work_dir}/train.log.json'
    cfg.env_cfg.log_level = 'INFO'
    cfg.log_processor = dict(type='LogProcessor', window_size=10, by_epoch=True)

    return cfg


if __name__ == "__main__":
    cfg = setup_cfg(data['config_path'], ast.literal_eval(data['classes']), data['load_from'], data['work_dir'])


    runner = Runner.from_cfg(cfg)

    runner.train()