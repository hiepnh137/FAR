sbatch --job-name=taxbjout \
       --output=./scratch/taxbjout.out \
       --error=./scratch/taxbjout.err \
       --nodes=1 \
       --gres=gpu:1 \
       --ntasks-per-node=1 \
       --cpus-per-task=16 \
       --mem=128gb \
       --time=1-00:00:00 \
       --sockets-per-node=1 \
       --cores-per-socket=8 \
       --qos=batch-short \
       --wrap="python test_far.py"

