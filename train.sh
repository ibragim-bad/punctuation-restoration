python src/train.py --cuda=True --pretrained-model=DeepPavlov/rubert-base-cased-conversational --freeze-bert=False --lstm-dim=-1 \
--language=ru --seed=1 --lr=5e-6 --epoch=10 --use-crf=False --augment-type=all  --augment-rate=0.15 \
--alpha-sub=0.4 --alpha-del=0.4 --data-path=data --save-path=out