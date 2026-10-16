# GazeTorch project

GazeTorch: Appearance-based Gaze Estimation (AGE) on PyTorch.

## Dependency
Minimal requirements:
* PyTorch >= 1.10
* OpenCV
* `timm` (for MAE models only)
* `imutils` (for glasses augmentation features)

## macOS webcam demo

See [demo/README.md](demo/README.md) for a local webcam-to-gaze-dot app with calibration, accuracy checks, and optional posture and drift checks.

## The ZeroGaze-10K dataset

https://drive.google.com/file/d/1TanMj9-NCEhAv9JAQteCh5wG1An9APSf/view

## The sample checkpoints

https://huggingface.co/7thChord/GazeTorch/tree/main


## Directory tree

* `config`
  * Stores config YAML proto & template config files.
  * `config_lib`
    * contains all config files in yaml format
  * `config_loader`
    * interface to extract the yaml file, main driver will be config_loader.py.
* `data`
  * Stores dataset classes
  * Dataset Preprocessing/Augmentation codes go here too
  * `dataset_loader.py` tool for loading dataset into the trainer
  * `augmenter` handles data augmentation & preprocessing.
* `model`
  * `backbone`, `fuser`, `head`
    * Stores existing backbone networks + modules
    * Backbones: **VGG**, **DenseNet**, **ResNet**, MobileNet_v2, CBAM, Coordinate Attention, **ViT**, BiSeNet
    * Feature fusers: Simple concat, Cross-Attention/Transformer, MBT
  * `net.py`: our basic `Net`(including TridentNet) class
    * it takes config file and construct a custom big module
  * `module_loader.py`: loader for subfolders (`backbone`, `fuser`, `head`)
  * `module_base.py` the base class that should be wrapping every submodule in `backbone`, `fuser`, `head`. Note that `loss` is a bit special and not required to be wrapped
* `tools`
  * All util functions go here
  * `cfgnode.py`: `CfgNode` class, a lightweight implementation of `Config_base`. Used in GazeTF project too.
  * `checkpoint.py`: tools for checkpointing, used by both trainer and net
  * `distributed.py`: tools for distributed computing, for example figure out who is the master process 
  * `printer.py`: printer function useful for figuring out which part of the printing is your information.
  * `param_constant.py`: preset identifiers used in codes and config files.
  * `utils/torch_ops.py`: utils tool for other stuff, *needs cleaning* 
* `train`
  * `loss` dir: stores loss functions
  * `loss_loader.py`: loader for `loss`
  * `optimizer_loader.py`: loader for optimizers.
  * `dataloader_wrapper.py`: `DataLoader_wrapper` class, an enhanced `DataLoader` that supports multi-view batching. Used in training.
  * `trainer.py`: `Trainer` class, the main interface to run training tasks.
  * `tester.py`: `Tester` class, a simplified `Trainer` to evaluate an existing ckpt on a target dataset.
  * `tester_tf.py`: `Tester` class specially for TFLite models.

## Run training
Example 1: Run training on all 8 GPUs
```bash
python -m torch.distributed.launch --nproc_per_node 8 gazetorch.py --cfg_path config/config_lib/alpha_mobilenet.yaml
```
or
```bash
torchrun --standalone --nodes=1 --nproc_per_node 8 gazetorch.py --cfg_path config/config_lib/alpha_mobilenet.yaml
```
Example 2: Run training on GPU #4-7
```bash
CUDA_VISIBLE_DEVICES="4,5,6,7" python -m torch.distributed.launch --nproc_per_node 4 --master_port 54345 gazetorch.py --cfg_path config/config_lib/alpha_mobilenet.yaml
```
or
```bash
CUDA_VISIBLE_DEVICES="4,5,6,7" torchrun --standalone --nnodes=1 --nproc_per_node 4 --master_port 54345 gazetorch.py --cfg_path config/config_lib/alpha_mobilenet.yaml
```
* `nproc_per_node` = # of GPUs to use.
* `CUDA_VISIBLE_DEVICES` = what GPUs to use.
* `master_port` is optional - to avoid probable conflicting training tasks
* `cfg_path` which cfg file to use

## Run evaluation

Example: batch evaluation on MPIIGazeFace
```bash
python gazetorch.py -m eval -f torch  \
    -d config/config_lib/eval/mpiiface.yaml -o temp_results/ \
    --load_ckpt_dir <path_to_the_pkl> \
    --result_module head --load_ckpt_name xxx.pkl
```

## Citation

If you find this work useful, please cite:

```bibtex
@misc{li2026realtimeappearancebasedgazeestimation,
      title={Real-time Appearance-based Gaze Estimation for Open Domains}, 
      author={Zhenhao Li and Zheng Liu and Seunghyun Lee and Amin Fadaeinejad and Yuanhao Yu},
      year={2026},
      eprint={2603.26945},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.26945}, 
}
