cd /users/smp23qw/dissertation/lycon
cat > run_rhyme300.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=rhyme300
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=rhyme300_%j.out
#SBATCH --error=rhyme300_%j.err

PY=/users/smp23qw/.conda/envs/lycon/bin/python
export HF_HOME=/mnt/parscratch/users/smp23qw/hf
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_XET=1

cd /users/smp23qw/dissertation/lycon

$PY src/generate_metrical.py --manifest prompts_300.jsonl --skeletons skeletons_300.jsonl --out-dir metrical_300_rhyme --n-candidates 8 --k-context 3 --no-syllable --anti-template --temperature 0.95 --w-count 0 --w-rhyme 3.0 --w-vocab 4.0 --w-rep 2.5 --resume --dump-candidates cand_dump_rhyme.jsonl
EOF
sbatch run_rhyme300.sh