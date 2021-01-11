
echo ${1}_morestats.csv
python ../util/log_visualizer.py --hide --export $1.png ${1}_log.json --morestats ${1}_morestats.csv
#cp $1.png ~/Dropbox/Osim
#aws s3 cp $1.png s3://opensimrl-wentao
cp $1.png ../wentaoplots
