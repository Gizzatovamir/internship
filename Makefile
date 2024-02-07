run: download_seqot_weights download_gem_weights local

run_docker: download_seqot_weights download_gem_weights docker

download_seqot_weights:
	poetry run gdown --id 1IY2jEdsd5tOPwSd_uaUrD2FdMfK7RD6X -O ./data/seqot.pth.tar

download_gem_weights:
	poetry run gdown --id 1UvKdwKT95TjuV5hCZFc8PRXb7H4tvIin -O ./data/gem.pth.tar

docker: docker_build docker_run

docker_build:
# 	docker build --no-cache -t seqot -f Docker/Dockerfile .
	docker build -t seqot -f Docker/Dockerfile .

docker_run:
	docker run -v data:/src/data --gpus "device=0" --name seqot_container seqot

local: gen_depth_data gen_sub_descriptors eval

gen_depth_data:
# args: --path <path to db>
	poetry run python gen_depth_data.py --path ./data/23_07_08_visual_odometry_0_0.db3
gen_sub_descriptors:
# args: --cfg <path to cfg>
	poetry run python gen_sub_descriptors.py --cfg config/config.yml
eval:
	poetry run python eval.py