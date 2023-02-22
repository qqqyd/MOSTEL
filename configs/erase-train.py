# Loss
lb = 1.
lb_mask = 1.
lb_beta = 10.

# Train
learning_rate = 1e-4
decay_rate = 0.9
beta1 = 0.9
beta2 = 0.999
max_iter = 100000
write_log_interval = 50
save_ckpt_interval = 50000
gen_example_interval = 50000
task_name = 'erase-train'
checkpoint_savedir = 'output/' + task_name + '/'  # dont forget '/'
ckpt_path = 'None'
vgg19_weights = 'models/vgg19-dcbb9e9d.pth'

# data
batch_size = 64
num_workers = 8
data_shape = [64, 256]
data_dir = ['datasets/training/EnsText-patch']
i_s_dir = 'i_s'
t_b_dir = 't_b'
mask_s_dir = 'mask_s'
example_data_dir = 'demo_img/imgs'
example_result_dir = checkpoint_savedir + 'val_visualization'

# predict
predict_ckpt_path = None
predict_data_dir = None
predict_result_dir = checkpoint_savedir + 'pred_result'
