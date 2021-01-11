for filename in *_out; do
    echo $filename
    tail -1 $filename
done
