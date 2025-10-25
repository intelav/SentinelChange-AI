import random

import ever as er
import numpy as np
import torch
from tqdm import tqdm

er.registry.register_all()


def register_leviscd_evaluate_fn(launcher):
    launcher.override_evaluate(evaluate_levircd)


def evaluate_levircd(self, test_dataloader, config=None):
    self.model.eval()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    det_metric_op = er.metric.PixelMetric(2,
                                          self.model_dir,
                                          logger=self.logger)

    with torch.no_grad():
        for img, ret_gt in tqdm(test_dataloader):
            img = img.to(device)

            y1y2change = self.model.module(img).sigmoid() > 0.5

            pr_change = y1y2change[:, 2, :, :].cpu()
            pr_change = pr_change.numpy().astype(np.uint8)
            gt_change = ret_gt['change']
            gt_change = gt_change.numpy()
            y_true = gt_change.ravel()
            y_pred = pr_change.ravel()

            y_true = np.where(y_true > 0, np.ones_like(y_true), np.zeros_like(y_true))

            det_metric_op.forward(y_true, y_pred)

    split = [s.replace('./LEVIR-CD/', '') for s in test_dataloader.config.root_dir]
    split_str = ','.join(split)
    self.logger.info(f'det -[LEVIRCD {split_str}]')
    det_metric_op.summary_all()

    torch.cuda.empty_cache()

try:
    from core import field
except ImportError:
    print("Warning: Could not import 'field' from core.field. Assuming default mask key.")
    class MockField:
        MASK1 = 'mask' # This is what xview2_dataset.py uses for the mask key
    field = MockField()


# Define the evaluation function specifically for xView2 building segmentation
def evaluate_xview2_building_segmentation(self, test_dataloader, config=None):
    self.model.eval()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # PixelMetric(2) is suitable for binary segmentation (building vs. non-building)
    metric_op = er.metric.PixelMetric(2,
                                      self.model_dir,
                                      logger=self.logger)

    with torch.no_grad():
        for img, ret_gt in tqdm(test_dataloader):
            img = img.to(device) # Input is a 3-channel image from xView2

            # The model's classifier head outputs the semantic mask.
            # Assuming self.model.module(img) directly returns the semantic prediction tensor
            predicted_mask_prob = self.model.module(img).sigmoid()

            # Threshold to get binary mask. Select the first channel (index 0).
            # The output should be [B, 1, H, W], so taking [:, 0, :, :] is correct.
            predicted_mask = (predicted_mask_prob[:, 0, :, :] > 0.5).float()

            pr_mask = predicted_mask.cpu().numpy().astype(np.uint8)
            # Ground truth is the building mask from xView2
            gt_mask = ret_gt[field.MASK1] # Access the mask using field.MASK1
            gt_mask = gt_mask.numpy()

            y_true = gt_mask.ravel()
            y_pred = pr_mask.ravel()

            # Ensure ground truth is binary (0 or 1)
            y_true = np.where(y_true > 0, np.ones_like(y_true), np.zeros_like(y_true))

            metric_op.forward(y_true, y_pred)

    # Log the dataset path correctly for xView2
    # You might need to adjust this logging if test_dataloader.config.image_dir is a tuple.
    if isinstance(test_dataloader.config.image_dir, tuple):
        split_str = ','.join([s.replace('/media/avaish/aiwork/satellite-work/datasets/xview2/', '') for s in test_dataloader.config.image_dir])
    else:
        split_str = test_dataloader.config.image_dir.replace('/media/avaish/aiwork/satellite-work/datasets/xview2/', '')

    self.logger.info(f'det -[xView2 Building Segmentation {split_str}]')
    metric_op.summary_all()

    torch.cuda.empty_cache()
if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    SEED = 2333
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.set_rng_state(torch.manual_seed(SEED).get_state())

    print("--- Debugging Data Loader ---")
    try:
        sample_img, sample_gt = next(iter(kw_dataloader['traindata_loader']))
        print(f"Train data loader - Image shape: {sample_img.shape}, GT shape: {sample_gt[field.MASK1].shape}")
        print(f"Train data loader - Image dtype: {sample_img.dtype}, GT dtype: {sample_gt[field.MASK1].dtype}")

        sample_eval_img, sample_eval_gt = next(iter(kw_dataloader['testdata_loader']))
        print(f"Test data loader - Image shape: {sample_eval_img.shape}, GT shape: {sample_eval_gt[field.MASK1].shape}")
        print(f"Test data loader - Image dtype: {sample_eval_img.dtype}, GT dtype: {sample_eval_gt[field.MASK1].dtype}")

    except Exception as e:
        print(f"Error inspecting data loader: {e}")
    print("---------------------------")

    trainer = er.trainer.get_trainer('th_amp_ddp')()
    #trainer.run(after_construct_launcher_callbacks=[register_leviscd_evaluate_fn])
    trainer.run(after_construct_launcher_callbacks=[
        lambda launcher: launcher.override_evaluate(evaluate_xview2_building_segmentation)
    ])