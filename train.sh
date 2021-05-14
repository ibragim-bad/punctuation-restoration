python src/new_ru_train.py --cuda=True --pretrained-model=DeepPavlov/rubert-base-cased --freeze-bert=False --lstm-dim=512 \
--language=ru --seed=1 --lr=1e-5 --epoch=11 --use-crf=False --augment-type=all  --augment-rate=0.01 --batch-size=20 \
--alpha-sub=0.4 --alpha-del=0.4 --data-path=data/ru/*.txt --save-path=out --sequence-length 256 # --yttm yttm.model --pqrnn=False #--hg True
# python src/new_ru_train.py --cuda=False --pretrained-model=DeepPavlov/rubert-base-cased- --freeze-bert=False --lstm-dim=-1 \
# --language=ru --seed=1 --lr=0.001 --epoch=20 --use-crf=False --augment-type=all  --augment-rate=0.0 --batch-size=256 \
# --alpha-sub=0.4 --alpha-del=0.4 --data-path=data/ru/*.txt  --save-path=out --sequence-length 92 --yttm yttm.model #--pqrnn=True