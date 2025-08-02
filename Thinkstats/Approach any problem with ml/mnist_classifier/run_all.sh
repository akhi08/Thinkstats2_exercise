# #!/bin/sh
# python ./src/train.py --fold 0
# python ./src/train.py --fold 1
# python ./src/train.py --fold 2
# python ./src/train.py --fold 3
# python ./src/train.py --fold 4



# ❯ sh run_all.sh
# updated after addition of model of dispatcher
# !/bin/sh
python ./src/train1.py --fold 0 --model random_forest
python ./src/train1.py --fold 1 --model random_forest
python ./src/train1.py --fold 2 --model random_forest
python ./src/train1.py --fold 3 --model random_forest
python ./src/train1.py --fold 4 --model random_forest