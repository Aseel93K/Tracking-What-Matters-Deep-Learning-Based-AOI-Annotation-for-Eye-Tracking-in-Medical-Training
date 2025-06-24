import torch
from typing import Dict, Optional, Union
from mmengine.hooks import Hook
from mmengine.runner import Runner, autocast
from mmdet.registry import HOOKS



import torch
import json
from mmengine.hooks import Hook
from typing import Optional, Dict, Union

@HOOKS.register_module()
class ValLoss(Hook):
    """Save and print valid loss info"""

    def __init__(self, loss_list=[], save_path='val_losses.json') -> None:
        self.loss_list = loss_list
        self.save_path = save_path

    def before_val(self, runner) -> None:
        self.model = runner.model

    def after_val_epoch(self, runner, metrics: Optional[Dict[str, float]] = None) -> None:
        if len(self.loss_list) > 0:
            loss_log = {}
            for lossInfo in self.loss_list:
                if 'loss' in lossInfo:
                    for tmp_loss_name, tmp_loss_value in lossInfo.items():

                        loss_log.setdefault(tmp_loss_name, []).append(tmp_loss_value.item())


            for loss_name, loss_values in loss_log.items():
                mean_loss = torch.mean(torch.tensor(loss_values))
                print(f'val/{loss_name}_val', mean_loss)
                runner.message_hub.update_scalar(f'val/{loss_name}_val', mean_loss)


            with open(self.save_path, 'a') as f:
                f.write(json.dumps(loss_log) + '\n')
        else:
            print('The model does not support validation loss!')

    @staticmethod
    def compute_loss(losses) -> dict:
        loss_dict = {}
        total_loss = 0.0

        for key, value in losses.items():
            if key.startswith('loss'):
                if isinstance(value, list):
                    loss_value = sum([v.item() for v in value])
                else:
                    loss_value = value.item()

                loss_dict[key] = loss_value
                total_loss += loss_value


        loss_dict['total_loss'] = total_loss

        return loss_dict

    def after_val_iter(self,
                       runner,
                       batch_idx: int,
                       data_batch: Union[dict, tuple, list] = None,
                       outputs: Optional[dict] = None) -> None:
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=runner.val_loop.fp16):
                data = self.model.data_preprocessor(data_batch, True)
                losses = self.model._run_forward(data, mode='loss')
                loss_dict = self.compute_loss(losses)
                runner.logger.info('Compute loss:!')
                log_data = {
                    "loss": loss_dict['total_loss'],
                    "loss_rpn_cls": loss_dict['loss_rpn_cls'],
                    "loss_rpn_bbox": loss_dict['loss_rpn_bbox'],
                    "loss_cls": loss_dict['loss_cls'],
                    "loss_bbox": loss_dict['loss_bbox'],
                    "epoch": runner.epoch,
                                   }

                log_file = f"{runner.log_dir}/validation_loss_log.json"
                with open(log_file, 'a') as f:
                    json.dump(log_data, f)
                    f.write('\n')  # Write each log entry on a new line
                self.loss_list.append(losses)
