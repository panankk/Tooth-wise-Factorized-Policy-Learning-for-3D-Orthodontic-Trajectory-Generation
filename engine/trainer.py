import torch
import os
import csv
import logging
import math
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class TrainingMonitor:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir=os.path.join(log_dir, 'tensorboard'))
        self.csv_path = os.path.join(log_dir, 'metrics.csv')
        self.csv_file = open(self.csv_path, 'w', newline='')

        self.headers = [
            'Epoch', 'Stage', 'Mode', 'Total',
            'Raw_P_MSE', 'Raw_P_NLL', 'Avg_Sigma_Pos',
            'Raw_R_MSE', 'Raw_R_NLL', 'Avg_Sigma_Rot', 'Var_Reg_Rot',
            'Raw_Focal', 'W_Focal', 'Raw_Pattern', 'W_Pattern', 'Raw_Bin', 'W_Bin',
            'Sigma_Focal', 'Sigma_Pattern', 'Sigma_Bin',
            'P_Acc', 'R_Acc', 'LR'
        ]
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(self.headers)

        self.logger = self._setup_console_logger()

        # Global best across all stages. This preserves your original behavior.
        self.best_loss = float('inf')

        # New: stage-wise best losses.
        # Because STAGE_1 / STAGE_2 / STAGE_3 use different objective forms,
        # their Total losses are not strictly comparable. These stage-wise
        # checkpoints let you evaluate the best checkpoint from each phase.
        self.best_stage_loss = {}

    def _setup_console_logger(self):
        logger = logging.getLogger('OrthoEngine')
        logger.setLevel(logging.INFO)

        # Keep original behavior: avoid duplicated handlers if logger already exists.
        if not logger.handlers:
            fh = logging.FileHandler(os.path.join(self.log_dir, 'train.log'))
            fh.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S'))

            sh = logging.StreamHandler()
            sh.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S'))

            logger.addHandler(fh)
            logger.addHandler(sh)

        return logger

    def log_epoch(self, epoch, stage_name, mode_name, avg, lr):
        self.writer.add_scalar('Epoch/Total', avg['Total'], epoch)

        row = [
            epoch, stage_name, mode_name, f"{avg['Total']:.6f}",
            f"{avg.get('Raw_P_MSE', 0):.6f}", f"{avg.get('Raw_P_NLL', 0):.6f}", f"{avg.get('Avg_Sigma_Pos', 0):.6f}",
            f"{avg.get('Raw_R_MSE', 0):.6f}", f"{avg.get('Raw_R_NLL', 0):.6f}", f"{avg.get('Avg_Sigma_Rot', 0):.6f}", f"{avg.get('Var_Reg_Rot', 0):.6f}",
            f"{avg.get('Raw_Focal', 0):.6f}", f"{avg.get('W_Focal', 0):.6f}",
            f"{avg.get('Raw_Pattern', 0):.6f}", f"{avg.get('W_Pattern', 0):.6f}",
            f"{avg.get('Raw_Bin', 0):.6f}", f"{avg.get('W_Bin', 0):.6f}",
            f"{avg.get('Sigma_Focal', 0):.6f}", f"{avg.get('Sigma_Pattern', 0):.6f}", f"{avg.get('Sigma_Bin', 0):.6f}",
            f"{avg.get('P_Acc', 0):.2f}", f"{avg.get('R_Acc', 0):.2f}", f"{lr:.2e}"
        ]
        self.csv_writer.writerow(row)
        self.csv_file.flush()

        msg = (
            f"Ep {epoch:03d} [{stage_name}|{mode_name}] | Total: {avg['Total']:.4f}\n"
            f"   [Pos] MSE:{avg.get('Raw_P_MSE', 0):.4f} NLL:{avg.get('Raw_P_NLL', 0):.4f} Sig:{avg.get('Avg_Sigma_Pos', 0):.3f}\n"
            f"   [Rot] MSE:{avg.get('Raw_R_MSE', 0):.4f} NLL:{avg.get('Raw_R_NLL', 0):.4f} Sig:{avg.get('Avg_Sigma_Rot', 0):.3f}\n"
            f"   [Stat] PAcc:{avg.get('P_Acc', 0):.1f}% | RAcc:{avg.get('R_Acc', 0):.1f}%"
        )
        self.logger.info(msg)

    def _make_save_dict(self, current_loss, model, criterion, epoch, stage_name):
        return {
            'model_state_dict': model.state_dict(),
            'criterion_state_dict': criterion.state_dict(),
            'epoch': epoch,
            'best_loss': current_loss,
            'stage': stage_name
        }

    def check_and_save_best(self, current_loss, model, criterion, epoch, save_dir, stage_name):
        """
        Save both:
        1. best_model_all_stages.pth: original global-best behavior.
        2. best_stage_*.pth: best checkpoint inside each training stage.

        This does not affect training, gradients, loss, optimizer, or scheduler.
        It only changes checkpoint saving.
        """

        # 1. Original global best across all stages.
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            save_dict = self._make_save_dict(current_loss, model, criterion, epoch, stage_name)

            save_path = os.path.join(save_dir, 'best_model_all_stages.pth')
            torch.save(save_dict, save_path)

            self.logger.info(
                f"🌟 New Global Best! Ep {epoch} ({stage_name}) Loss: {current_loss:.4f} -> best_model_all_stages.pth"
            )

        # 2. New stage-wise best.
        prev_stage_best = self.best_stage_loss.get(stage_name, float('inf'))

        if current_loss < prev_stage_best:
            self.best_stage_loss[stage_name] = current_loss
            save_dict = self._make_save_dict(current_loss, model, criterion, epoch, stage_name)

            safe_stage_name = stage_name.lower()
            save_name = f"best_{safe_stage_name}.pth"
            save_path = os.path.join(save_dir, save_name)

            torch.save(save_dict, save_path)

            self.logger.info(
                f"🏅 New Stage Best! Ep {epoch} ({stage_name}) Loss: {current_loss:.4f} -> {save_name}"
            )

    def close(self):
        self.writer.close()
        self.csv_file.close()


class Trainer:
    def __init__(self, model, dataloader, criterion, optimizer, scheduler, device, config):
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config

        self.monitor = TrainingMonitor(config.SAVE_DIR)

    def get_stage_info(self, epoch):
        cfg = self.config

        if epoch < cfg.STAGE1_EPOCHS:
            return "STAGE_1", "BASE_REC"
        elif epoch < cfg.STAGE2_EPOCHS:
            return "STAGE_2", "POS_UNCERT"
        else:
            rot_epoch = epoch - cfg.STAGE2_EPOCHS
            if rot_epoch < cfg.STAGE3_FREEZE_EPOCHS:
                return "STAGE_3_FREEZE", "ROT_FROZEN"
            else:
                return "STAGE_3_NORMAL", "ROT_FULL"

    def train_epoch(self, epoch, stage_name, mode_name):
        self.model.train()
        self.criterion.set_stage(stage_name)

        pbar = tqdm(
            enumerate(self.dataloader),
            total=len(self.dataloader),
            desc=f"Ep {epoch} [{mode_name}]"
        )

        # Initialize all stats.
        acc_metrics = {k: 0.0 for k in self.monitor.headers[3:-1]}

        valid_batches = 0

        for i, batch in pbar:
            # 1. Unpack & move to device.
            shape = batch['shape'].to(self.device)
            state = batch['input_seq'].view(-1, 32, 18).to(self.device)
            timestep = batch['timestep'].view(-1).to(self.device)
            mask = batch['teeth_mask'].view(-1, 32).to(self.device)
            tooth_types = batch['tooth_types'].view(-1, 32).to(self.device)

            strat_vec_pos = batch['strat_vec_pos'].view(-1, 32, 10).to(self.device)
            strat_vec_rot = batch['strat_vec_rot'].view(-1, 32, 10).to(self.device)
            feat_prev_pos = batch['feat_prev_pos'].view(-1, 32).to(self.device)
            feat_prev_rot = batch['feat_prev_rot'].view(-1, 32).to(self.device)

            gt_pos_mu = batch['gt_pos_mu'].view(-1, 32, 3).to(self.device)
            gt_rot_mu = batch['gt_rot_mu'].view(-1, 32, 6).to(self.device)
            gt_mask_pos = batch['gt_mask_pos'].view(-1, 32).to(self.device)
            gt_mask_rot = batch['gt_mask_rot'].view(-1, 32).to(self.device)

            # 2. Forward model.
            mu_pos, log_var_pos, mu_rot, log_var_rot, logits_pos, logits_rot = self.model(
                shape,
                state,
                timestep,
                tooth_types,
                mask,
                strat_vec_pos=strat_vec_pos,
                strat_vec_rot=strat_vec_rot
            )

            # 3. Forward loss.
            loss, d = self.criterion(
                mu_pos,
                log_var_pos,
                mu_rot,
                log_var_rot,
                logits_pos,
                logits_rot,
                gt_pos_mu,
                gt_rot_mu,
                gt_mask_pos,
                gt_mask_rot,
                mask,
                feat_prev_pos,
                feat_prev_rot
            )

            if not torch.isfinite(loss):
                self.monitor.logger.warning(f"⚠️ NaN Loss at Ep {epoch}, Step {i}. Skipping.")
                continue

            # 4. Backward & optimize.
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.criterion.parameters()),
                1.0
            )
            self.optimizer.step()
            self.scheduler.step()

            valid_batches += 1

            # 5. Track stats.
            for k in acc_metrics:
                if k in d and isinstance(d[k], (int, float)) and math.isfinite(d[k]):
                    acc_metrics[k] += d[k]

            if i % 10 == 0:
                pbar.set_postfix({
                    'L': f"{d.get('Total', 0):.2f}",
                    'SigP': f"{d.get('Avg_Sigma_Pos', 0):.3f}"
                })

        denom = max(valid_batches, 1)
        avg = {k: v / denom for k, v in acc_metrics.items()}
        return avg

    def train(self, start_epoch):
        self.monitor.logger.info("🚀 Starting Training Engine")

        for epoch in range(start_epoch, self.config.NUM_EPOCHS):
            # Dynamic stage routing.
            stage_name, mode_name = self.get_stage_info(epoch)

            if epoch == self.config.STAGE1_EPOCHS:
                self.monitor.logger.info("➡️ Entering STAGE 2: Unlocking Position Uncertainty")
            elif epoch == self.config.STAGE2_EPOCHS:
                self.monitor.logger.info("➡️ Entering STAGE 3A: Unlocking Rotation NLL (Frozen Variance)")
            elif epoch == self.config.STAGE2_EPOCHS + self.config.STAGE3_FREEZE_EPOCHS:
                self.monitor.logger.info("➡️ Entering STAGE 3B: Unfreezing Rotation Variance")

            # Train one epoch.
            avg_stats = self.train_epoch(epoch + 1, stage_name, mode_name)

            # Log & save.
            lr = self.optimizer.param_groups[0]['lr']
            self.monitor.log_epoch(epoch + 1, stage_name, mode_name, avg_stats, lr)

            self.monitor.check_and_save_best(
                avg_stats['Total'],
                self.model,
                self.criterion,
                epoch + 1,
                self.config.SAVE_DIR,
                stage_name
            )

            # Regular interval checkpoint.
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'criterion_state_dict': self.criterion.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'stage': stage_name
                }, os.path.join(self.config.SAVE_DIR, f"checkpoint_ep{epoch + 1}.pth"))

        self.monitor.close()
        self.monitor.logger.info("🏁 Engine Shutdown: Training Finished.")
