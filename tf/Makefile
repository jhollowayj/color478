all:
	python colorize.py

notebook:
	ipython notebook --ip='*'

clean:
	rm -f val-imgs/*
	rm -f gen-train.csv gen-val.csv
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
