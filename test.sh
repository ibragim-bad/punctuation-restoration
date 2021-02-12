# python3 src/inference.py --pretrained-model=albert-base-v2 --weight-path=weights.pt --language=en --cuda=false \
# --in-file=data/test_en.txt --out-file=data/test_en_out.txt
# python3 src/inference.py --pretrained-model=xlm-roberta-large --language=en --cuda=false \
# --in-file=data/test_en.txt --out-file=data/test_en_out.txt
# python3 src/inference.py --pretrained-model=prajjwal1/bert-small --language=en --cuda=false \
# --in-file=data/test_en.txt --out-file=data/test_en_out.txt
# python3 src/inference.py --pretrained-model=DeepPavlov/rubert-base-cased-conversational --weight-path=out/weights.pt --language=ru --cuda=false \
# --in-file=data/test_ru.txt --out-file=data/test_ru_out.txt
python3 src/inference.py --pretrained-model=sberbank-ai/sbert_large_nlu_ru --weight-path=out/weights.pt --language=ru --cuda=true \
--in-file=data/test_ru.txt --out-file=data/test_ru_out.txt  --sequence-length 92