class trainer(object):
    def __init__(self, optimizer, optimizer_params, lr_scheduler, lr_scheduler_params, model):
        self.optimier = optimizer
        self.optimier_params = optimizer_params
        self.lr_schedueler = lr_scheduler
        self.lr_schedueler_params = lr_scheduler_params
        self.model = model