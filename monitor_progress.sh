while true;do
    find . -name "*.json" -cmin -5|while read fname; do
        echo $fname
        python util/log_visualizer.py --hide --export $fname.png $fname
        #aws s3 cp $fname.png s3://opensimrl-wentao
        cp $fname.png wentaoplots
    done
    sleep 300
done
