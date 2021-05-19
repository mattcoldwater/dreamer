## Run:  
1. replace GAME with boxing, pong .....  
2. Remember to change vm2drive.sh  
3. (Use tiny mode): from models.tiny_observation import ObservationDecoder, ObservationEncoder (agent.py:11)  

## Resume Run:  
1. get weights from, for example "GAME_yin2_0.tgz"  
%cd '/gdrive/My Drive/CSE525/Project/wu/scripts/'  
!bash drive2vm.sh "GAME_yin2_0"  
%cd '/content/'  
2. Remember to add CumSteps, find "CumSteps" from "Diagnostics/CumSteps"!  

yin2:  
python main.py --log-dir /content/GAME_yin2_0/ --game GAME --cuda-idx 0  
python main.py --log-dir /content/GAME_yin2_1/ --game GAME --cuda-idx 0 --load-model-path /content/GAME_yin2_0/run_2/params.pkl --CumSteps **  

yin3:  
python main.py --log-dir /content/GAME_small_0/ --game GAME --n_parallel 4 --cuda-idx 0  
python main.py --log-dir /content/GAME_small_1/ --game GAME --n_parallel 4 --cuda-idx 0 --load-model-path ** --CumSteps **;  

yin3 V2:  
python main.py --log-dir /content/GAME_small_0_v2/ --game GAME --cuda-idx 0 --n_parallel **;  
python main.py --log-dir /content/GAME_small_1_v2/ --game GAME --cuda-idx 0 --load-model-path /content/GAME_small_0_v2/run_0/params.pkl --CumSteps ** --n_parallel **   

## For simple train&test:
python main.py --n_parallel 1 --cuda-idx 0  
python main.py --n_parallel 1 --cuda-idx 0 --eval --load-model-path /content/data/local/20210519/run_0/params.pkl

## Other info:
--log-dir /content/boxing_yin2_1/ --game boxing --cuda-idx 0 --load-model-path /content/boxing_yin2_0/run_2/params.pkl --CumSteps 150000  
--log-dir /content/boxing_small_2_v2/ --game boxing --cuda-idx 0 --load-model-path /content/boxing_small_1_v2/run_0/params.pkl --CumSteps 390000
  
boxing_yin2_0 CumSteps=150000  
boxing_small_0_v2 CumSteps=240000   
boxing_small_1_v2 CumSteps=240000 + (200000-5e4)=390000    

## References:
1. https://github.com/juliusfrost/dreamer-pytorch
2. https://github.com/danijar/dreamer
3. https://github.com/danijar/dreamerv2
4. https://github.com/google-research/dreamer
5. Paper: https://arxiv.org/abs/1912.01603
