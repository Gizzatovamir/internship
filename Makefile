gen_depth_data:
# args: --path <path to db>
	poetry run python src/gen_depth_data.py ${ARGS}
gen_sub_descriptors:
# args: --cfg <path to cfg>
	poetry run python src/gen_sub_descriptors.py ${ARGS}
eval:
	poetry run python src/eval.py