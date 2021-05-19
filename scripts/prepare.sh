pip install torch==1.5.0 torchvision==0.6.0
pip install --user git+https://github.com/astooke/rlpyt.git
pip install pyprind

# git config --global user.email "mattcoldwater@gmail.com"
# git config --global user.name "mattcoldwater"

# DO NOT RUN requires.txt! no need for COLAB!!

# import psutil
# import os
# print(psutil.Process(os.getpid()).memory_full_info().uss / (1024.**3))

# algo = Dreamer(horizon=10, kl_scale=0.1, use_pcont=True, initial_optim_state_dict=optimizer_state_dict, replay_size=int(5e5))