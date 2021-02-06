# python3 src/inference.py --pretrained-model=albert-base-v2 --weight-path=weights.pt --language=en --cuda=false \
# --in-file=data/test_en.txt --out-file=data/test_en_out.txt
# python3 src/inference.py --pretrained-model=xlm-roberta-large --language=en --cuda=false \
# --in-file=data/test_en.txt --out-file=data/test_en_out.txt
# python3 src/inference.py --pretrained-model=prajjwal1/bert-small --language=en --cuda=false \
# --in-file=data/test_en.txt --out-file=data/test_en_out.txt
python3 src/inference.py --pretrained-model=bert-base-uncased --language=en --cuda=false \
--in-file=data/test_en.txt --out-file=data/test_en_out.txt