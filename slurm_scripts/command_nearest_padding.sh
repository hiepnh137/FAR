sbatch --job-name=Tax16 \
       --output=./scratch/TaxiBJout16.out \
       --error=./scratch/TaxiBJout16.err \
       --nodes=1 \
       --gres=gpu:1 \
       --ntasks-per-node=1 \
       --cpus-per-task=16 \
       --mem=128gb \
       --time=3-00:00:00 \
       --sockets-per-node=1 \
       --cores-per-socket=8 \
       --qos=batch-short \
       --wrap="python test_far_spatio_temporal_padding.py --data TaxiBJout_16_6 --seq_len 16 --pred_len 6"

sbatch --job-name=Taxi32 \
       --output=./scratch/TaxiBJout32.out \
       --error=./scratch/TaxiBJout32.err \
       --nodes=1 \
       --gres=gpu:1 \
       --ntasks-per-node=1 \
       --cpus-per-task=16 \
       --mem=128gb \
       --time=3-00:00:00 \
       --sockets-per-node=1 \
       --cores-per-socket=8 \
       --qos=batch-short \
       --wrap="python test_far_spatio_temporal_padding.py --data TaxiBJout_32_6 --seq_len 32 --pred_len 6"


sbatch --job-name=Tax64 \
       --output=./scratch/TaxiBJout64.out \
       --error=./scratch/TaxiBJout64.err \
       --nodes=1 \
       --gres=gpu:1 \
       --ntasks-per-node=1 \
       --cpus-per-task=16 \
       --mem=128gb \
       --time=3-00:00:00 \
       --sockets-per-node=1 \
       --cores-per-socket=8 \
       --qos=batch-short \
       --wrap="python test_far_spatio_temporal_padding.py --data TaxiBJout_64_6 --seq_len 64 --pred_len 6"

sbatch --job-name=Tax128 \
       --output=./scratch/TaxiBJout128.out \
       --error=./scratch/TaxiBJout128.err \
       --nodes=1 \
       --gres=gpu:1 \
       --ntasks-per-node=1 \
       --cpus-per-task=16 \
       --mem=128gb \
       --time=3-00:00:00 \
       --sockets-per-node=1 \
       --cores-per-socket=8 \
       --qos=batch-short \
       --wrap="python test_far_spatio_temporal_padding.py --data TaxiBJout_128_6 --seq_len 128 --pred_len 6"



