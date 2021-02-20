python src/distillation.py --cuda=True --pretrained-model=DeepPavlov/rubert-base-cased-conversational --freeze-bert=False --lstm-dim=-1 \
--language=ru --seed=1 --lr=5e-5 --epoch=201 --use-crf=False --augment-type=all  --augment-rate=0.15 --batch-size=32 --sequence-length 128 \
--alpha-sub=0.4 --alpha-del=0.4 --data-path=data/ru/**/**/*.txt --save-path=out