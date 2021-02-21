if [ "$1" != "no-preprocess" ]; then
    
    # BPE

    TASK=couplet
    
    # preprocess

    fairseq-preprocess \
      --source-lang "src" \
      --target-lang "dst" \
      --trainpref "./dataset/train" \
      --testpref "./dataset/test" \
      --validpref "./dataset/valid" \
      --destdir "${TASK}-bin/" \
      --joined-dictionary \
      --workers 60 \
    
fi

# Training

CUDA_VISIBLE_DEVICES=0 fairseq-train ${TASK}-bin/ \
    --lr 3e-5 --clip-norm 0.1 --dropout 0.1 --max-tokens 1024 \
    --lr-scheduler polynomial_decay --total-num-update 20000 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --arch transformer --save-dir './checkpoints/transformer/' --optimizer adam \
    --tensorboard-logdir "./logs/tensorboard/transformer/" \
    --skip-invalid-size-inputs-valid-test \
    --fp16
