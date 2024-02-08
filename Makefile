run: download_seqot_weights download_gem_weights local

run_docker: docker

download_seqot_weights:
	poetry run gdown --id 1IY2jEdsd5tOPwSd_uaUrD2FdMfK7RD6X -O /data/seqot.pth.tar

download_gem_weights:
	poetry run gdown --id 1UvKdwKT95TjuV5hCZFc8PRXb7H4tvIin -O /data/gem.pth.tar

docker: docker_build docker_run

docker_build:
# 	docker build --no-cache -t seqot -f Docker/Dockerfile .
	docker build -t seqot -f Docker/Dockerfile .

docker_run:
	docker run --rm -v data:/data -v to_test:/to_test --gpus "device=0" --name seqot_container seqot

local: download_seqot_weights download_gem_weights gen_depth_data gen_sub_descriptors eval

gen_depth_data:
# args: --path <path to db>
	poetry run python gen_depth_data.py --path /data/23_07_08_visual_odometry_0_0.db3 --dst_path /data/depth_data
gen_sub_descriptors:
# args: --cfg <path to cfg>
	poetry run python gen_sub_descriptors.py --cfg config/config.yml --dst_path /data/
eval:
	poetry run python eval.py --cfg config/config.yaml

