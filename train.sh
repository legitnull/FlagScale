rm /share/project/fengyupu/github/FlagScale/outputs/PI0_BASE/logs/host_0_localhost.output
python run.py \
	--config-path ./examples/pi0_base/conf \
	--config-name train \
	action=run
