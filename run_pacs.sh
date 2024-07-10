for dataset in 'PACS'
do
    for target in 'art_painting' 'photo' 'cartoon' 'sketch'; do
        for rec_weight in 0.1; do
	        sbatch --job-name=$dataset-$target batch.sh $dataset $target $rec_weight
        done
    done
done