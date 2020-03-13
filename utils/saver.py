import os
import torch

class Saver():
    def __init__(self, save_dir, max_files=10):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        self.log_list = []
        self.save_dir = save_dir
        self.max_files = max_files
        self.saver_log_path = os.path.join(save_dir, '.saver_log')
        if os.path.isfile(self.saver_log_path):
            with open(self.saver_log_path, 'r') as f:
                self.log_list = f.read().splitlines()


    def save_checkpoint(self, model, epoch, ckpt_name, best=False):
        if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()

        state = {'epoch': epoch, 'model_state': model_state}
        ckpt_name = '{}.pth'.format(ckpt_name)
        save_path = os.path.join(self.save_dir, ckpt_name)
        torch.save(state, save_path)

        self.log_list.insert(0, save_path)
        if len(self.log_list) > self.max_files:
            pop_file = self.log_list.pop()
            if pop_file != save_path:
                if os.path.isfile(pop_file):
                    os.remove(pop_file)

        with open(self.saver_log_path, 'w') as f:
            for log in self.log_list:
                f.write(log + '\n')
        
    def load_checkpoint(self, model, filename):
        if os.path.isfile(filename):
            log_str("==> Loading from checkpoint %s" % filename)
            checkpoint = torch.load(filename)
            epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model_state'])
            log_str("==> Done")
        else:
            raise FileNotFoundError

        return epoch