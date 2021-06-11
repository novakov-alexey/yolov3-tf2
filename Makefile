train:
	python train.py \
		--dataset ./data/open-images-v6-train-800/tf.records \
		--val_dataset ./data/open-images-v6-validation-200/tf.records \
		--classes ./data/open-images-v6.names \
		--num_classes 1 \
		--mode fit --transfer no_output \
		--batch_size 16 \
		--epochs 10 \
		--weights ./checkpoints/yolov3.tf \
		--weights_num_classes 80 

visualize:
	python tools/visualize_dataset.py \
		--classes=data/open-images-v6.names \
		--dataset=data/open-images-v6-validation-100/tf.records

infer:
	python detect.py \
		--classes=data/open-images-v6.names \
		--num_classes=1 \
		--weights=./checkpoints/yolov3_train_4.tf \
		--image=./data/test-toy.jpg

load-datasets:
	python fiftyone_util.py load_datasets

convert-2-tf-records:
	python fiftyone_util.py convert