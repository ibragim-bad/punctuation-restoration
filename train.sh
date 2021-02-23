python src/new_ru_train.py --cuda=True --pretrained-model=DeepPavlov/rubert-base-cased --freeze-bert=False --lstm-dim=-1 \
--language=ru --seed=1 --lr=2e-5 --epoch=22 --use-crf=False --augment-type=all  --augment-rate=0.05 --batch-size=64 \
--alpha-sub=0.4 --alpha-del=0.4 --data-path=data/ru/*.txt --save-path=out --sequence-length 128 --yttm yttm.model --pqrnn=True #--hg True
# python src/new_ru_train.py --cuda=False --pretrained-model=DeepPavlov/rubert-base-cased- --freeze-bert=False --lstm-dim=-1 \
# --language=ru --seed=1 --lr=0.001 --epoch=20 --use-crf=False --augment-type=all  --augment-rate=0.0 --batch-size=256 \
# --alpha-sub=0.4 --alpha-del=0.4 --data-path=data/ru/*.txt  --save-path=out --sequence-length 92 --yttm yttm.model #--pqrnn=True