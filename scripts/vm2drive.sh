# python main.py --log-dir /content/pong_0/ --game pong --save-model 'gap'
exp_name="boxing_small_1_NoEps"
exp_path="boxing_small_1_v2/run_0" 

while :
do
    cd "/content/"
    tar -cf "${exp_name}.tgz" "./${exp_path}/"
    cp "${exp_name}.tgz" "/gdrive/My Drive/CSE525/Project/logs/"
    echo "transfer ${exp_name}.tgz done"
    sleep 600
done