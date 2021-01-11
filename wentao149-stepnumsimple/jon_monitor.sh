#!/bin/bash
if [[ $# -le 0 ]] ; then
    echo './monitor.sh progress_file morestats_file sleep_interval'
    exit 1
fi

PROGRESS_FILE=${1}_log.json
MORESTATS_FILE=${1}_morestats.csv
SLEEP_INTERVAL_SECS=60
THIS_DIR=${PWD##*/} 

echo 'Use the following to launch a webserver: sudo python3 -m http.server 1234'

while true;
do
	mkdir -p ../wentaoplots/$THIS_DIR
	# cp $PROGRESS_FILE ../jon-www/$THIS_DIR/log.json
	#python ../util/concatcsv.py models/progress.csv models/experiment_2017_08_14_10_08_17_910401_SGT_13ef2/progress.csv www/progress.csv
	# ./view ../www-stats/$THIS_DIR/progress.csv --hide --export ../www-stats/$THIS_DIR/progress.png
    OUTPUT="$(python ../util/log_visualizer.py $PROGRESS_FILE --morestats $MORESTATS_FILE --hide --export ../wentaoplots/$THIS_DIR/progress.png)"
    echo $OUTPUT
	echo "<img src=\"progress.png\" style=\"max-width:99%; max-height:99%;\" /><br />Last Updated: `date` <br />${OUTPUT}<br /> <script type=\"text/javascript\">setTimeout(function(){location = ''},30000)</script>" > ../wentaoplots/$THIS_DIR/index.html
	sleep $SLEEP_INTERVAL_SECS;
done
