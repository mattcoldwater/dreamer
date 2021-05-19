exp_name=$1
echo "${exp_name}"

cd "/content/"
cp "/gdrive/My Drive/CSE525/Project/logs/${exp_name}.tgz" "./${exp_name}.tgz"
tar -xf "${exp_name}.tgz" -C "./"