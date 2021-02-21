 fairseq-interactive couplet-bin/ \
    --results-path ./generate-result/ \
    --source-lang "src" \
    --target-lang "dst" \
    --path checkpoints/checkpoint_best.pt \
    --beam 5 \