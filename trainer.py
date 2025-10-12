import torch
import torch.optim as optim
import math
import util

class HeteroTrainer:
    """
    Trainer for models handling heterogeneous graphs.
    Supports multiple adjacency matrices, curriculum learning, 
    gradient clipping, and inverse scaling.
    """
    def __init__(self, model, lrate, wdecay, clip, step_size, seq_out_len, scaler, device, cl=True):
        self.model = model.to(device)
        self.scaler = scaler
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss_fn = util.masked_mae
        self.clip = clip
        self.step_size = step_size
        self.seq_out_len = seq_out_len
        self.curriculum = cl
        self.iter = 1
        self.task_level = 1

    def train(self, x, y, adjs=None):
        """
        x: input tensor (batch, seq_len, num_nodes, features)
        y: target tensor (batch, seq_len, num_nodes, features)
        adjs: dictionary of adjacency matrices for heterogeneous graph
        """
        self.model.train()
        self.optimizer.zero_grad()

        if adjs is not None:
            output = self.model(x.to(self.device), adjs={k: v.to(self.device) for k, v in adjs.items()})
        else:
            output = self.model(x.to(self.device))

        output = output.transpose(1, 3)  # adjust shape: (batch, features, num_nodes, seq_len)
        y_true = torch.unsqueeze(y.to(self.device), dim=1)
        y_pred = self.scaler.inverse_transform(output)

        # Curriculum learning
        if self.iter % self.step_size == 0 and self.task_level < self.seq_out_len:
            self.task_level += 1

        if self.curriculum:
            loss = self.loss_fn(y_pred[:, :, :, :self.task_level], y_true[:, :, :, :self.task_level], 0.0)
        else:
            loss = self.loss_fn(y_pred, y_true, 0.0)

        loss.backward()

        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

        self.optimizer.step()

        mape = util.masked_mape(y_pred, y_true, 0.0).item()
        rmse = util.masked_rmse(y_pred, y_true, 0.0).item()

        self.iter += 1
        return loss.item(), mape, rmse

    def eval(self, x, y, adjs=None):
        """
        Evaluation on validation/test data.
        """
        self.model.eval()
        with torch.no_grad():
            if adjs is not None:
                output = self.model(x.to(self.device), adjs={k: v.to(self.device) for k, v in adjs.items()})
            else:
                output = self.model(x.to(self.device))

            output = output.transpose(1, 3)
            y_true = torch.unsqueeze(y.to(self.device), dim=1)
            y_pred = self.scaler.inverse_transform(output)

            loss = self.loss_fn(y_pred, y_true, 0.0)
            mape = util.masked_mape(y_pred, y_true, 0.0).item()
            rmse = util.masked_rmse(y_pred, y_true, 0.0).item()

        return loss.item(), mape, rmse


class HeteroOptim:
    """
    Optimizer wrapper supporting gradient clipping and learning rate decay.
    """
    def __init__(self, params, method, lr, clip, lr_decay=1, start_decay_at=None):
        self.params = params
        self.lr = lr
        self.clip = clip
        self.method = method
        self.lr_decay = lr_decay
        self.start_decay_at = start_decay_at
        self.start_decay = False
        self.last_ppl = None
        self._make_optimizer()

    def _make_optimizer(self):
        if self.method == 'sgd':
            self.optimizer = optim.SGD(self.params, lr=self.lr, weight_decay=self.lr_decay)
        elif self.method == 'adagrad':
            self.optimizer = optim.Adagrad(self.params, lr=self.lr, weight_decay=self.lr_decay)
        elif self.method == 'adadelta':
            self.optimizer = optim.Adadelta(self.params, lr=self.lr, weight_decay=self.lr_decay)
        elif self.method == 'adam':
            self.optimizer = optim.Adam(self.params, lr=self.lr, weight_decay=self.lr_decay)
        else:
            raise RuntimeError("Invalid optimizer method: " + self.method)

    def step(self):
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.params, self.clip)
        self.optimizer.step()

    def update_learning_rate(self, ppl, epoch):
        if self.start_decay_at is not None and epoch >= self.start_decay_at:
            self.start_decay = True
        if self.last_ppl is not None and ppl > self.last_ppl:
            self.start_decay = True

        if self.start_decay:
            self.lr = self.lr * self.lr_decay
            print(f"Decaying learning rate to {self.lr}")
        self.start_decay = False
        self.last_ppl = ppl
        self._make_optimizer()
