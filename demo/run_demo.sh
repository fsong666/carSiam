source /home/sf/anaconda3/bin/activate pysot

video=street.mp4
#video=lion.mp4
#video=Elephants.mp4
#video=dance.mp4
#video=cave.mp4
#video=bag.avi

path=../experiments/siamcar_r50

#model=det_35_e19.pth
#model=vid_40_e19.pth
#model=det_vid_e19.pth
model=ods_det_vid_e19.pth
#model=general_model.pth


python ../tools/demo.py \
	--config ${path}/config.yaml \
	--snapshot ${path}/${model}  \
	--video_name ${video}