python src/new_ru_train.py --cuda=True --pretrained-model=DeepPavlov/rubert-base-cased-conversational --freeze-bert=False --lstm-dim=-1 \
--language=ru --seed=1 --lr=5e-5 --epoch=3 --use-crf=False --augment-type=all  --augment-rate=0.15 --batch-size=32 \
--alpha-sub=0.4 --alpha-del=0.4 --data-path=data/ru/**/**/*.txt --save-path=out
# python src/train.py --cuda=True --pretrained-model=sberbank-ai/sbert_large_nlu_ru --freeze-bert=False --lstm-dim=-1 \
# --language=ru --seed=1 --lr=5e-5 --epoch=3 --use-crf=False --augment-type=all  --augment-rate=0.15 --batch-size=12 \
# --alpha-sub=0.4 --alpha-del=0.4 --data-path=data --save-path=out --sequence-length 92