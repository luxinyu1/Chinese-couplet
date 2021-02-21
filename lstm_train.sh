if [ "$1" != "no-preprocess" ]; then
    
    # BPE

    TASK=couplet
    
    # preprocess

    fairseq-preprocess \
      --source-lang "src" \
      --target-lang "dst" \
      --trainpref "./dataset/train" \
      --testpref "./dataset/test" \
      --validpref "./dataset/test" \
      --destdir "${TASK}-bin/" \
      --joined-dictionary \
      --workers 60 \
    
fi

# Training

TOTAL_NUM_UPDATES=20000
WARMUP_UPDATES=500
MAX_TOKENS=1024
UPDATE_FREQ=1
LR=3e-05
    
CUDA_VISIBLE_DEVICES=0 fairseq-train ${TASK}-bin/ \
    --source-lang "src" \
    --target-lang "dst" \
    --arch lstm --save-dir ./checkpoints \
    --dropout 0.1 \
    --optimizer adam --lr 0.005 \
    --validate-interval 1 \
    --max-tokens 12000 \
    --fp16