all:
	python colorize.py
fast: 
	python colorize.py --batch_size 25 --train_data ../../color478/images/train.txt --val_data ../../color478/images/val.txt --learning_rate 0.25
me: 
	python colorize.py --batch_size 4 --train ../../color478/images/train.txt --val ../../color478/images/val.txt >output.txt &
bwmon:
	python colorize.py --batch_size 4 --train ../../color478/images/train.txt --val ../../color478/images/bwmonson.txt >output.txt &
check:
	python colorize.py --save True --forward shark.jpg
notebook:
	ipython notebook --ip='*'

clean:
	rm -f val-imgs/*
	rm -f train-loss.csv
	rm -f val-loss.csv
	rm -f *checkpoint*	
	rm -f log/*
	rm -rf out.jpg

model-clean:
	rm -f colorize.tfmodel

colorize.tfmodel:

DATE=$(shell date +%Y%m%d)
pkg:
	-mkdir colorize-$(DATE)
	python colorize.py --forward shark.jpg --save
	mv colorize.tfmodel colorize-$(DATE)
	cp shark.jpg forward.py colorize-$(DATE)
	tar -czvf colorize-$(DATE).tgz colorize-$(DATE)
	aws s3 cp colorize-$(DATE).tgz s3://tinyclouds-storage  --acl public-read
