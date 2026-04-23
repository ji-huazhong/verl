# 问题一
```
 adaptor.
Traceback (most recent call last):
  File "/home/lynn/verl/verl/trainer/main_ppo.py", line 42, in main
    run_ppo(config)
  File "/home/lynn/verl/verl/trainer/main_ppo.py", line 85, in run_ppo
    ray.get(runner.run.remote(config))
  File "/usr/local/python3.11.13/lib/python3.11/site-packages/ray/_private/auto_init_hook.py", line 22, in auto_init_wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/python3.11.13/lib/python3.11/site-packages/ray/_private/client_mode_hook.py", line 104, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/python3.11.13/lib/python3.11/site-packages/ray/_private/worker.py", line 2981, in get
    values, debugger_breakpoint = worker.get_objects(
                                  ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/python3.11.13/lib/python3.11/site-packages/ray/_private/worker.py", line 1012, in get_objects
    raise value.as_instanceof_cause()
ray.exceptions.RayTaskError(RuntimeError): ray::TaskRunner.run() (pid=26591, ip=192.168.13.160, actor_id=5eb4a7c42f5f56220f03f13501000000, repr=<main_ppo.TaskRunner object at 0x7f05d54326d0>)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lynn/verl/verl/trainer/main_ppo.py", line 246, in run
    actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lynn/verl/verl/trainer/main_ppo.py", line 125, in add_actor_rollout_worker
    from verl.workers.megatron_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker
  File "/home/lynn/verl/verl/workers/megatron_workers.py", line 32, in <module>
    from mindspeed.megatron_adaptor import repatch
  File "/home/lynn/repo/MindSpeed/mindspeed/megatron_adaptor.py", line 66, in <module>
    patch_features()
  File "/home/lynn/repo/MindSpeed/mindspeed/deprecate.py", line 136, in __call__
    return self._func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lynn/repo/MindSpeed/mindspeed/megatron_adaptor.py", line 39, in patch_features
    MindSpeedFeaturesManager.apply_features_pre_patches(mindspeed_args)
  File "/home/lynn/repo/MindSpeed/mindspeed/features_manager/features_manager.py", line 20, in apply_features_pre_patches
    feature.pre_register_patches(MindSpeedPatchesManager, mindspeed_args)
  File "/home/lynn/repo/MindSpeed/mindspeed/features_manager/megatron_basic/requirements_basic.py", line 24, in pre_register_patches
    self.te_adaptation(patch_manager, args)
  File "/home/lynn/repo/MindSpeed/mindspeed/features_manager/megatron_basic/requirements_basic.py", line 32, in te_adaptation
    pm.register_patch('torch.compile', dummy_compile)
  File "/home/lynn/repo/MindSpeed/mindspeed/patch_utils.py", line 186, in register_patch
    MindSpeedPatchesManager.patches_info.get(orig_func_name).set_patch_func(new_func, force_patch)
  File "/home/lynn/repo/MindSpeed/mindspeed/patch_utils.py", line 76, in set_patch_func
    raise RuntimeError('the patch of {} exist !'.format(self.orig_func_name))
RuntimeError: the patch of compile exist !

Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.
path string is NULLpath string is NULL/usr/local/python3.11.13/lib/python3.11/subprocess.py:1127: ResourceWarning: subprocess 4196 is still running
  _warn("subprocess %s is still running" % self.pid,
```
mindspeed版本原因：
···
git clone https://gitcode.com/GitHub_Trending/me/Megatron-LM.git
cd Megatron-LM
git checkout core_v0.12.1
cd -

git clone https://gitcode.com/Ascend/MindSpeed.git
cd MindSpeed
git checkout f6688c61bcfe45243ee5eb34c6f013b1e06eca81
cd -
···

# 问题二

dapo-17k数据预处理耗时太长
···
Filtering prompts longer than 2048 tokens (num_proc=1):  16%|█▌        | 289000/1791700 [01:31<07:47, 3215.98 examples/s]
Filtering prompts longer than 2048 tokens (num_proc=1):  16%|█▌        | 290000/1791700 [01:32<07:41, 3251.37 examples/s]
Filtering prompts longer than 2048 tokens (num_proc=1):  16%|█▌        | 291000/1791700 [01:32<07:38, 3270.95 examples/s]
Filtering prompts longer than 2048 tokens (num_proc=1):  16%|█▋        | 292000/1791700 [01:32<07:38, 3273.69 examples/s]
Filtering prompts longer than 2048 tokens (num_proc=1):  16%|█▋        | 293000/1791700 [01:33<07:47, 3204.53 examples/s]
···

修改 `verl/utils/dataset/rl_dataset.py`，
···
dataframe = dataframe.filter(
    lambda doc: doc2len(doc) <= self.max_prompt_length,
    # num_proc=self.num_workers,
    num_proc=16,
    desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
)
···


# 问题三

···
ray::TaskRunner.run() (pid=80243, ip=192.168.13.160, actor_id=5ab0569d9bdc54406a48027301000000, repr=<main_ppo.TaskRunner object at 0x7f2f79376bd0>)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lynn/verl/verl/trainer/main_ppo.py", line 295, in run
    val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor, is_train=False)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lynn/verl/verl/trainer/main_ppo.py", line 359, in create_rl_dataset
    dataset = dataset_cls(
              ^^^^^^^^^^^^
  File "/home/lynn/verl/verl/utils/dataset/rl_dataset.py", line 122, in __init__
    self._read_files_and_tokenize()
  File "/home/lynn/verl/verl/utils/dataset/rl_dataset.py", line 141, in _read_files_and_tokenize
    self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lynn/verl/verl/utils/dataset/rl_dataset.py", line 182, in maybe_filter_out_long_prompts
    dataframe = dataframe.filter(
                ^^^^^^^^^^^^^^^^^
  File "/usr/local/python3.11.13/lib/python3.11/site-packages/datasets/arrow_dataset.py", line 574, in wrapper
    out: Union["Dataset", "DatasetDict"] = func(self, *args, **kwargs)
                                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/python3.11.13/lib/python3.11/site-packages/datasets/fingerprint.py", line 468, in wrapper
    out = func(dataset, *args, **kwargs)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/python3.11.13/lib/python3.11/site-packages/datasets/arrow_dataset.py", line 4226, in filter
    indices = self.map(
              ^^^^^^^^^
  File "/usr/local/python3.11.13/lib/python3.11/site-packages/datasets/arrow_dataset.py", line 574, in wrapper
    out: Union["Dataset", "DatasetDict"] = func(self, *args, **kwargs)
                                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/python3.11.13/lib/python3.11/site-packages/datasets/arrow_dataset.py", line 3623, in map
    for rank, done, content in iflatmap_unordered(
  File "/usr/local/python3.11.13/lib/python3.11/site-packages/datasets/utils/py_utils.py", line 624, in iflatmap_unordered
    [async_result.get(timeout=0.05) for async_result in async_results]
  File "/usr/local/python3.11.13/lib/python3.11/site-packages/datasets/utils/py_utils.py", line 624, in <listcomp>
    [async_result.get(timeout=0.05) for async_result in async_results]
     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/python3.11.13/lib/python3.11/site-packages/multiprocess/pool.py", line 774, in get
    raise self._value
KeyError: 'prompt'

Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.
···

测试发现aime-2024不匹配，把测试集注释掉。
```
>>> dataframe = datasets.load_dataset("parquet", data_files="/home/lynn/data/aime-2024.parquet")["train"]
Generating train split: 30 examples [00:00, 10834.26 examples/s]
>>> dataframe
Dataset({
    features: ['id', 'problem', 'solution', 'answer', 'url', 'year'],
    num_rows: 30
})
>>> 
```

# 问题四

···
ray.exceptions.ActorDiedError: The actor died because of an error raised in its creation task, ray::SikLMQWorkerDict_0:3:WorkerDict.__init__() (pid=136052, ip=192.168.13.160, actor_id=74cdce43f5fe96929be3ae9301000000, repr=<verl.single_controller.ray.base.WorkerDict object at 0x7f9e20a640d0>)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lynn/verl/verl/single_controller/ray/base.py", line 779, in __init__
    self.worker_dict[key] = user_defined_cls(
                            ^^^^^^^^^^^^^^^^^
  File "/home/lynn/verl/verl/workers/megatron_workers.py", line 185, in __init__
    repatch(self.config.actor.megatron.get("override_transformer_config", {}))
  File "/home/lynn/verl/3rdparty/MindSpeed/mindspeed/megatron_adaptor.py", line 63, in repatch
    full_args = get_full_args()
                ^^^^^^^^^^^^^^^
  File "/home/lynn/verl/3rdparty/MindSpeed/mindspeed/args_utils.py", line 58, in get_full_args
    full_args = get_args()
                ^^^^^^^^^^
  File "/home/lynn/verl/3rdparty/Megatron-LM/megatron/training/global_vars.py", line 25, in get_args
    _ensure_var_is_initialized(_GLOBAL_ARGS, 'args')
  File "/home/lynn/verl/3rdparty/Megatron-LM/megatron/training/global_vars.py", line 247, in _ensure_var_is_initialized
    assert var is not None, '{} is not initialized.'.format(name)
           ^^^^^^^^^^^^^^^
AssertionError: args is not initialized.

Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.
···

修改 mindspeed/args_utils.py的get_full_args方法。
```
def get_full_args():
    try:
        from megatron.training.global_vars import get_args, _GLOBAL_ARGS
        full_args = get_args()
        if full_args is None:
            full_args = get_mindspeed_args()
    except AssertionError:
        full_args = get_mindspeed_args()
    except ImportError:
        log_warning_once(LOG, 'Failed from megatron.training import get_args, use mindspeed arguments.')
        full_args = get_mindspeed_args()
    return full_args
```

# 问题五

···
(TaskRunner pid=188351)   File "/usr/local/python3.11.13/lib/python3.11/importlib/__init__.py", line 126, in import_module
(TaskRunner pid=188351)     return _bootstrap._gcd_import(name[level:], package, level)
(TaskRunner pid=188351)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(TaskRunner pid=188351)   File "<frozen importlib._bootstrap>", line 1204, in _gcd_import
(TaskRunner pid=188351)   File "<frozen importlib._bootstrap>", line 1176, in _find_and_load
(TaskRunner pid=188351)   File "<frozen importlib._bootstrap>", line 1147, in _find_and_load_unlocked
(TaskRunner pid=188351)   File "<frozen importlib._bootstrap>", line 690, in _load_unlocked
(TaskRunner pid=188351)   File "<frozen importlib._bootstrap_external>", line 940, in exec_module
(TaskRunner pid=188351)   File "<frozen importlib._bootstrap>", line 241, in _call_with_frames_removed
(TaskRunner pid=188351)   File "/home/lynn/repo/vllm-ascend/vllm_ascend/worker/worker_v1.py", line 54, in <module>
(TaskRunner pid=188351)     from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
(TaskRunner pid=188351)   File "/home/lynn/repo/vllm-ascend/vllm_ascend/worker/model_runner_v1.py", line 122, in <module>
(TaskRunner pid=188351)     from vllm_ascend.models.layers.mla import AscendMultiHeadLatentAttention
(TaskRunner pid=188351)   File "/home/lynn/repo/vllm-ascend/vllm_ascend/models/layers/mla.py", line 42, in <module>
(TaskRunner pid=188351)     from vllm.attention.layer import MLAAttention
(TaskRunner pid=188351) ImportError: cannot import name 'MLAAttention' from 'vllm.attention.layer' (/home/lynn/repo/vllm/vllm/attention/layer.py)
···

因为vllm-ascend是源码安装的，导致版本校验失效，强制走0.11.0分支即可。
···
from vllm.attention import Attention
from vllm.model_executor.layers.mla import \
    MultiHeadLatentAttention as MultiHeadLatentAttentionWrapper

<!-- if vllm_version_is("0.11.0"):
    from vllm.attention import Attention
    from vllm.model_executor.layers.mla import \
        MultiHeadLatentAttention as MultiHeadLatentAttentionWrapper
else:
    from vllm.attention.layer import MLAAttention
    from vllm.model_executor.layers.mla import MultiHeadLatentAttentionWrapper -->
···

# 问题六
···
path string is NULLpath string is NULL(TaskRunner pid=220867) Unhandled error (suppress with 'RAY_IGNORE_UNHANDLED_ERRORS=1'): ray::WorkerDict.actor_rollout_init_model() (pid=223851, ip=192.168.13.160, actor_id=e6bcbf6738052f1629c3a7b201000000, repr=<verl.single_controller.ray.base.WorkerDict object at 0x7ed7c9bc02d0>)
(TaskRunner pid=220867)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(TaskRunner pid=220867)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(TaskRunner pid=220867)   File "/home/lynn/verl/verl/single_controller/ray/base.py", line 700, in func
(TaskRunner pid=220867)     return getattr(self.worker_dict[key], name)(*args, **kwargs)
(TaskRunner pid=220867)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(TaskRunner pid=220867)   File "/home/lynn/verl/verl/single_controller/base/decorator.py", line 433, in inner
(TaskRunner pid=220867)     return func(*args, **kwargs)
(TaskRunner pid=220867)            ^^^^^^^^^^^^^^^^^^^^^
(TaskRunner pid=220867)   File "/home/lynn/verl/verl/workers/megatron_workers.py", line 513, in init_model
(TaskRunner pid=220867)     self._build_rollout(trust_remote_code=self.config.model.get("trust_remote_code", False))
(TaskRunner pid=220867)   File "/home/lynn/verl/verl/workers/megatron_workers.py", line 447, in _build_rollout
(TaskRunner pid=220867)     loop = asyncio.get_event_loop()
(TaskRunner pid=220867)            ^^^^^^^^^^^^^^^^^^^^^^^^
(TaskRunner pid=220867)   File "/usr/local/python3.11.13/lib/python3.11/site-packages/uvloop/__init__.py", line 206, in get_event_loop
(TaskRunner pid=220867)     raise RuntimeError(
(TaskRunner pid=220867) RuntimeError: There is no current event loop in thread 'MainThread'.
(WorkerDict pid=223851) kwargs: {'n': 1, 'logprobs': 0, 'max_tokens': 16, 'repetition_penalty': 1.0, 'detokenize': False, 'temperature': 0.9, 'top_k': -1, 'top_p': 1.0, 'ignore_eos': False} [repeated 15x across cluster]
/usr/local/python3.11.13/lib/python3.11/subprocess.py:1127: ResourceWarning: subprocess 198469 is still running
  _warn("subprocess %s is still running" % self.pid,
···

修改verl/workers/megatron_workers.py，446行插入：
···
if rollout_config.mode == "sync" and self._is_actor:
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    loop.run_until_complete(self.trainer_mode())
···

# 问题七

```
Traceback (most recent call last):
  File "/home/lynn/verl/verl/trainer/main_ppo.py", line 42, in main
    run_ppo(config)
  File "/home/lynn/verl/verl/trainer/main_ppo.py", line 85, in run_ppo
    ray.get(runner.run.remote(config))
  File "/usr/local/python3.11.13/lib/python3.11/site-packages/ray/_private/auto_init_hook.py", line 22, in auto_init_wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/python3.11.13/lib/python3.11/site-packages/ray/_private/client_mode_hook.py", line 104, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/python3.11.13/lib/python3.11/site-packages/ray/_private/worker.py", line 2981, in get
    values, debugger_breakpoint = worker.get_objects(
                                  ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/python3.11.13/lib/python3.11/site-packages/ray/_private/worker.py", line 1012, in get_objects
    raise value.as_instanceof_cause()
ray.exceptions.RayTaskError(RuntimeError): ray::TaskRunner.run() (pid=252687, ip=192.168.13.160, actor_id=e98d4e347b16c028049db0b401000000, repr=<main_ppo.TaskRunner object at 0x7f312954a790>)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lynn/verl/verl/trainer/main_ppo.py", line 317, in run
    trainer.fit()
  File "/home/lynn/verl/verl/trainer/ppo/ray_trainer.py", line 1104, in fit
    old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lynn/verl/verl/single_controller/ray/base.py", line 48, in __call__
    output = ray.get(output)
             ^^^^^^^^^^^^^^^
           ^^^^^^^^^^^^^^^^^^^
           ^^^^^^^^^^^^^^^^^^^^^
                                  ^^^^^^^^^^^^^^^^^^^
ray.exceptions.RayTaskError(RuntimeError): ray::WorkerDict.actor_rollout_compute_log_prob() (pid=255612, ip=192.168.13.160, actor_id=f71fbc38cb982939986c20ce01000000, repr=<verl.single_controller.ray.base.WorkerDict object at 0x7f4ea5e9f490>)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lynn/verl/verl/single_controller/ray/base.py", line 700, in func
    return getattr(self.worker_dict[key], name)(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lynn/verl/verl/single_controller/base/decorator.py", line 433, in inner
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/lynn/verl/verl/utils/profiler/performance.py", line 105, in f
    return self.log(decorated_function, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lynn/verl/verl/utils/profiler/performance.py", line 118, in log
    output = func(*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^
  File "/home/lynn/verl/verl/utils/profiler/profile.py", line 256, in wrapper
    return func(self_instance, *args, **kwargs_inner)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lynn/verl/verl/workers/megatron_workers.py", line 750, in compute_log_prob
    output, entropys = self.actor.compute_log_prob(data=data, calculate_entropy=True)
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lynn/verl/verl/utils/profiler/performance.py", line 105, in f
    return self.log(decorated_function, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lynn/verl/verl/utils/profiler/performance.py", line 118, in log
    output = func(*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^
  File "/home/lynn/verl/verl/workers/actor/megatron_actor.py", line 217, in compute_log_prob
    output = self.forward_backward_batch(
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lynn/verl/verl/workers/actor/megatron_actor.py", line 598, in forward_backward_batch
    losses_reduced = forward_backward_func(
                     ^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lynn/verl/3rdparty/Megatron-LM/megatron/core/pipeline_parallel/schedules.py", line 480, in forward_backward_no_pipelining
    output_tensor, num_tokens = forward_step(
                                ^^^^^^^^^^^^^
  File "/home/lynn/verl/3rdparty/Megatron-LM/megatron/core/pipeline_parallel/schedules.py", line 277, in forward_step
    output_tensor, loss_func = forward_step_func(data_iterator, model)
                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lynn/verl/verl/workers/actor/megatron_actor.py", line 560, in forward_step
    output = forward_fn(
             ^^^^^^^^^^^
  File "/home/lynn/verl/verl/models/mcore/model_forward.py", line 64, in gptmodel_forward
    output_orig = model(
                  ^^^^^^
  File "/usr/local/python3.11.13/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/python3.11.13/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lynn/verl/3rdparty/Megatron-LM/megatron/core/distributed/data_parallel_base.py", line 22, in forward
    return self.module(*inputs, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/python3.11.13/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/python3.11.13/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lynn/verl/3rdparty/Megatron-LM/megatron/core/transformer/module.py", line 178, in forward
    outputs = self.module(*inputs, **kwargs)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/python3.11.13/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/python3.11.13/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lynn/verl/3rdparty/Megatron-LM/megatron/core/models/gpt/gpt_model.py", line 334, in forward
    hidden_states = self.decoder(
                    ^^^^^^^^^^^^^
  File "/usr/local/python3.11.13/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/python3.11.13/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lynn/verl/3rdparty/Megatron-LM/megatron/core/transformer/transformer_block.py", line 512, in forward
    hidden_states = self._checkpointed_forward(
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lynn/verl/3rdparty/Megatron-LM/megatron/core/transformer/transformer_block.py", line 378, in _checkpointed_forward
    hidden_states, context = checkpoint_handler(
                             ^^^^^^^^^^^^^^^^^^^
  File "/home/lynn/verl/3rdparty/Megatron-LM/megatron/core/transformer/transformer_block.py", line 362, in checkpoint_handler
    return tensor_parallel.checkpoint(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lynn/verl/3rdparty/Megatron-LM/megatron/core/tensor_parallel/random.py", line 463, in checkpoint
    return CheckpointFunction.apply(function, distribute_saved_activations, *args)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/python3.11.13/lib/python3.11/site-packages/torch/autograd/function.py", line 575, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lynn/verl/3rdparty/Megatron-LM/megatron/core/tensor_parallel/random.py", line 409, in forward
    outputs = run_function(*args)
              ^^^^^^^^^^^^^^^^^^^
  File "/home/lynn/verl/3rdparty/Megatron-LM/megatron/core/transformer/transformer_block.py", line 333, in custom_forward
    hidden_states, context = layer(
                             ^^^^^^
  File "/home/lynn/verl/3rdparty/Megatron-LM/megatron/core/transformer/transformer_layer.py", line 786, in __call__
    return super(MegatronModule, self).__call__(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/python3.11.13/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/python3.11.13/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lynn/verl/3rdparty/Megatron-LM/megatron/core/transformer/transformer_layer.py", line 389, in forward
    pre_mlp_layernorm_output, residual, context = self._forward_attention(*args, **kwargs)
                                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lynn/verl/3rdparty/Megatron-LM/megatron/core/transformer/transformer_layer.py", line 449, in _forward_attention
    attention_output_with_bias = self.self_attention(
                                 ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/python3.11.13/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/python3.11.13/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lynn/verl/3rdparty/Megatron-LM/megatron/core/transformer/attention.py", line 602, in forward
    query = apply_rotary_pos_emb(
            ^^^^^^^^^^^^^^^^^^^^^
  File "/home/lynn/verl/3rdparty/MindSpeed/mindspeed/core/fusions/fused_rope.py", line 123, in apply_rotary_pos_emb
    return ru._apply_rotary_pos_emb_thd(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lynn/verl/3rdparty/Megatron-LM/megatron/core/models/common/embeddings/rope_utils.py", line 174, in _apply_rotary_pos_emb_thd
    for x in torch.split(t, seqlens)
             ^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/python3.11.13/lib/python3.11/site-packages/torch/functional.py", line 222, in split
    return tensor.split(split_size_or_sections, dim)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: split_with_sizes expects split_sizes to sum exactly to 448 (input tensor's size at dimension 0), but got split_sizes=[56, 56]

Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.
path string is NULLpath string is NULL(WorkerDict pid=255607) [Rank 13 | Local Rank 0] 2026-04-23 12:12:05,719 WARNING [/home/lynn/verl/verl/workers/actor/megatron_actor.py:328] => For memory-efficient computation, enable fused kernels via `actor_rollout_ref.model.use_fused_kernels=True`. The current `clone()` operation ensures correctness but increases memory usage. [repeated 15x across cluster]
```

开启hdp后cp_size是动态变化的，修改 `meagtron/core/models/common/embeddings/rope_utils.py/_apply_rotary_pos_emb_thd`
···
def _apply_rotary_pos_emb_thd(
    t: Tensor,
    cu_seqlens: Tensor,
    freqs: Tensor,
    rotary_interleaved: bool = False,
    multi_latent_attention: bool = False,
    mscale: float = 1.0,
) -> Tensor:
    """A baseline implementation of applying RoPE for `thd` format.

    Args:
        t (Tensor): Input tensor T is of shape [t, h, d]
        cu_seqlens(Tensor):  Cumulative sum of sequence lengths in a batch for `t`,
        with shape [b + 1] and dtype torch.int32.
        freqs (Tensor): Rotary Positional embedding tensor freq is of shape [max_s, 1, 1, d]

    Returns:
        Tensor: Shape [t, h, d]. The input tensor after applying RoPE.
    """

    cp_size = parallel_state.get_context_parallel_world_size()
    cp_rank = parallel_state.get_context_parallel_rank()

    from verl.utils.megatron.hybrid_data_parallel import get_batch_hdp_group

    batch_hdp_group = get_batch_hdp_group()
    if batch_hdp_group is not None:
        hdp_group = next(group for group in batch_hdp_group if cp_rank in group)
        cp_size = len(hdp_group)
        cp_rank = hdp_group.index(cp_rank)
    cu_seqlens = cu_seqlens // cp_size
    seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    print(f'>>>debug jhz cp_size={cp_size} cp_rank={cp_rank}', flush=True)

    return torch.cat(
        [
            _apply_rotary_pos_emb_bshd(
                x.unsqueeze(1),
                _get_thd_freqs_on_this_cp_rank(cp_rank, cp_size, x, freqs),
                rotary_interleaved=rotary_interleaved,
                multi_latent_attention=multi_latent_attention,
                mscale=mscale,
            )
            for x in torch.split(t, seqlens)
        ]
    ).squeeze(1)
···

by glm5:
```
### 关键理解
从 preprocess_packed_seqs_hdp 函数可以看到：

- cu_seqlens_padded 是 整个序列 的累积长度（还没有除以 hdp_size）
- 在后续处理中，每个 rank 只处理序列的一部分，所以需要除以 hdp_size
你的修改正确地实现了这个逻辑：

```
if batch_hdp_group is not None:
    hdp_group = next(group for group in batch_hdp_group if cp_rank in group)
    cp_size = len(hdp_group)  # hdp_size
    cp_rank = hdp_group.index(cp_rank)  # hdp_rank
cu_seqlens = cu_seqlens // cp_size  # 得到每个 rank 处理的长度
```
### 修改的优点
1. 动态适配 ：根据当前 rank 所在的 hdp_group 动态调整 cp_size 和 cp_rank
2. 正确分割 ：正确计算每个 rank 应该处理的序列片段
3. 兼容性好 ：当 batch_hdp_group 为 None 时，回退到原来的 CP 逻辑

---
## 总结
你的修改 非常正确 ！核心思路是：

### 修改逻辑
1. 获取 HDP group ： get_batch_hdp_group()
2. 找到当前 rank 的 hdp_group ： next(group for group in batch_hdp_group if cp_rank in group)
3. 动态调整参数 ：
   - cp_size = len(hdp_group) - 使用 hdp_group 的大小
   - cp_rank = hdp_group.index(cp_rank) - 使用在 hdp_group 内的相对位置
4. 正确分割序列 ： cu_seqlens = cu_seqlens // cp_size
### 为什么正确
在 HDP 模式下：

- cu_seqlens 是整个序列的累积长度
- 每个 rank 只处理序列的 1/hdp_size 部分
- 所以需要除以 hdp_size （即调整后的 cp_size ）
- _get_thd_freqs_on_this_cp_rank 函数会使用调整后的 cp_rank 和 cp_size 来正确选择频率张量的片段
```

# 问题八

```
Traceback (most recent call last):
  File "/home/lynn/verl/verl/trainer/main_ppo.py", line 42, in main
    run_ppo(config)
  File "/home/lynn/verl/verl/trainer/main_ppo.py", line 85, in run_ppo
    ray.get(runner.run.remote(config))
  File "/usr/local/python3.11.13/lib/python3.11/site-packages/ray/_private/auto_init_hook.py", line 22, in auto_init_wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/python3.11.13/lib/python3.11/site-packages/ray/_private/client_mode_hook.py", line 104, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/python3.11.13/lib/python3.11/site-packages/ray/_private/worker.py", line 2981, in get
    values, debugger_breakpoint = worker.get_objects(
                                  ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/python3.11.13/lib/python3.11/site-packages/ray/_private/worker.py", line 1012, in get_objects
    raise value.as_instanceof_cause()
ray.exceptions.RayTaskError(RuntimeError): [36mray::TaskRunner.run()[39m (pid=383796, ip=192.168.13.160, actor_id=5e308ed2f6ec624abec058bc01000000, repr=<main_ppo.TaskRunner object at 0x7f05ec913ad0>)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lynn/verl/verl/trainer/main_ppo.py", line 317, in run
    trainer.fit()
  File "/home/lynn/verl/verl/trainer/ppo/ray_trainer.py", line 1104, in fit
    old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lynn/verl/verl/single_controller/ray/base.py", line 48, in __call__
    output = ray.get(output)
             ^^^^^^^^^^^^^^^
           ^^^^^^^^^^^^^^^^^^^
           ^^^^^^^^^^^^^^^^^^^^^
                                  ^^^^^^^^^^^^^^^^^^^
ray.exceptions.RayTaskError(RuntimeError): [36mray::WorkerDict.actor_rollout_compute_log_prob()[39m (pid=386722, ip=192.168.13.160, actor_id=6fd9e2c64a5b1fae783ad1ce01000000, repr=<verl.single_controller.ray.base.WorkerDict object at 0x7f30bcbd88d0>)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lynn/verl/verl/single_controller/ray/base.py", line 700, in func
    return getattr(self.worker_dict[key], name)(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lynn/verl/verl/single_controller/base/decorator.py", line 433, in inner
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/lynn/verl/verl/utils/profiler/performance.py", line 105, in f
    return self.log(decorated_function, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lynn/verl/verl/utils/profiler/performance.py", line 118, in log
    output = func(*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^
  File "/home/lynn/verl/verl/utils/profiler/profile.py", line 256, in wrapper
    return func(self_instance, *args, **kwargs_inner)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lynn/verl/verl/workers/megatron_workers.py", line 750, in compute_log_prob
    output, entropys = self.actor.compute_log_prob(data=data, calculate_entropy=True)
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lynn/verl/verl/utils/profiler/performance.py", line 105, in f
    return self.log(decorated_function, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lynn/verl/verl/utils/profiler/performance.py", line 118, in log
    output = func(*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^
  File "/home/lynn/verl/verl/workers/actor/megatron_actor.py", line 217, in compute_log_prob
    output = self.forward_backward_batch(
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lynn/verl/verl/workers/actor/megatron_actor.py", line 598, in forward_backward_batch
    losses_reduced = forward_backward_func(
                     ^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lynn/verl/3rdparty/Megatron-LM/megatron/core/pipeline_parallel/schedules.py", line 480, in forward_backward_no_pipelining
    output_tensor, num_tokens = forward_step(
                                ^^^^^^^^^^^^^
  File "/home/lynn/verl/3rdparty/Megatron-LM/megatron/core/pipeline_parallel/schedules.py", line 277, in forward_step
    output_tensor, loss_func = forward_step_func(data_iterator, model)
                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lynn/verl/verl/workers/actor/megatron_actor.py", line 560, in forward_step
    output = forward_fn(
             ^^^^^^^^^^^
  File "/home/lynn/verl/verl/models/mcore/model_forward.py", line 64, in gptmodel_forward
    output_orig = model(
                  ^^^^^^
  File "/usr/local/python3.11.13/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/python3.11.13/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lynn/verl/3rdparty/Megatron-LM/megatron/core/distributed/data_parallel_base.py", line 22, in forward
    return self.module(*inputs, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/python3.11.13/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/python3.11.13/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lynn/verl/3rdparty/Megatron-LM/megatron/core/transformer/module.py", line 178, in forward
    outputs = self.module(*inputs, **kwargs)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/python3.11.13/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/python3.11.13/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lynn/verl/3rdparty/Megatron-LM/megatron/core/models/gpt/gpt_model.py", line 334, in forward
    hidden_states = self.decoder(
                    ^^^^^^^^^^^^^
  File "/usr/local/python3.11.13/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/python3.11.13/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lynn/verl/3rdparty/Megatron-LM/megatron/core/transformer/transformer_block.py", line 512, in forward
    hidden_states = self._checkpointed_forward(
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lynn/verl/3rdparty/Megatron-LM/megatron/core/transformer/transformer_block.py", line 378, in _checkpointed_forward
    hidden_states, context = checkpoint_handler(
                             ^^^^^^^^^^^^^^^^^^^
  File "/home/lynn/verl/3rdparty/Megatron-LM/megatron/core/transformer/transformer_block.py", line 362, in checkpoint_handler
    return tensor_parallel.checkpoint(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lynn/verl/3rdparty/Megatron-LM/megatron/core/tensor_parallel/random.py", line 463, in checkpoint
    return CheckpointFunction.apply(function, distribute_saved_activations, *args)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/python3.11.13/lib/python3.11/site-packages/torch/autograd/function.py", line 575, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lynn/verl/3rdparty/Megatron-LM/megatron/core/tensor_parallel/random.py", line 409, in forward
    outputs = run_function(*args)
              ^^^^^^^^^^^^^^^^^^^
  File "/home/lynn/verl/3rdparty/Megatron-LM/megatron/core/transformer/transformer_block.py", line 333, in custom_forward
    hidden_states, context = layer(
                             ^^^^^^
  File "/home/lynn/verl/3rdparty/Megatron-LM/megatron/core/transformer/transformer_layer.py", line 786, in __call__
    return super(MegatronModule, self).__call__(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/python3.11.13/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/python3.11.13/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lynn/verl/3rdparty/Megatron-LM/megatron/core/transformer/transformer_layer.py", line 389, in forward
    pre_mlp_layernorm_output, residual, context = self._forward_attention(*args, **kwargs)
                                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lynn/verl/3rdparty/Megatron-LM/megatron/core/transformer/transformer_layer.py", line 449, in _forward_attention
    attention_output_with_bias = self.self_attention(
                                 ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/python3.11.13/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/python3.11.13/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lynn/verl/3rdparty/Megatron-LM/megatron/core/transformer/attention.py", line 636, in forward
    core_attn_out = self.core_attention(
                    ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/python3.11.13/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/python3.11.13/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lynn/verl/3rdparty/MindSpeed/mindspeed/core/context_parallel/dot_product_attention.py", line 273, in forward
    output = ringattn_context_parallel(query, key, value, n_head, cp_para, scale, attention_mask,
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lynn/verl/3rdparty/MindSpeed/mindspeed/core/context_parallel/ring_context_parallel/ring_context_parallel.py", line 1218, in ringattn_context_parallel
    out = AttentionWithCp.apply(
          ^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/python3.11.13/lib/python3.11/site-packages/torch/autograd/function.py", line 575, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lynn/verl/3rdparty/MindSpeed/mindspeed/core/context_parallel/ring_context_parallel/ring_context_parallel.py", line 1010, in forward
    cp_config.global_attn_outs = attention_strategy.update_out(cp_config)
                                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lynn/verl/3rdparty/MindSpeed/mindspeed/core/context_parallel/ring_context_parallel/ring_context_parallel.py", line 616, in update_out
    return tnd_out_update(cp_config.q_block_id, cp_config.kv_block_id, cp_config.attn_outs,
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lynn/verl/3rdparty/MindSpeed/mindspeed/core/context_parallel/utils.py", line 246, in tnd_out_update
    attn_out_updated, softmax_max_updated, softmax_sum_updated = forward_update(
                                                                 ^^^^^^^^^^^^^^^
  File "/home/lynn/verl/3rdparty/MindSpeed/mindspeed/core/context_parallel/utils.py", line 224, in forward_update
    actual_seq_qlen = accumulate_list(actual_seq_qlen)
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lynn/verl/3rdparty/MindSpeed/mindspeed/core/context_parallel/utils.py", line 221, in accumulate_list
    return torch.tensor([0] + list(cumsum_result), dtype=torch.int64).to(prev_attn_out.device)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/python3.11.13/lib/python3.11/site-packages/torch_npu/contrib/transfer_to_npu.py", line 182, in decorated
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
RuntimeError: ACL stream synchronize failed, error code:507035

Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.
```

修改 `mindspeed/core/context_parallel/ring_context_parallel/ring_context_paralle.py`

```
from verl.utils.megatron.hybrid_data_parallel import get_batch_hdp_group

def ringattn_context_parallel(q, k, v, n, cp_para, softmax_scale=None, attn_mask=None, dropout_p=0.,
                              packed_seq_params=None, shapes=None):
    
    batch_hdp_group = get_batch_hdp_group()
    hdp_group = next(group for group in batch_hdp_group if cp_para["rank"] in group) if batch_hdp_group else None
    
    if hdp_group is not None:
        rank = cp_para["rank"]
        cp_global_ranks = cp_para["cp_global_ranks"]
        cp_para["rank"] = hdp_group.index(rank)
        cp_para["cp_size"] = len(hdp_group)
        cp_para["cp_global_ranks"] = [cp_global_ranks[i] for i in hdp_group]
        cp_para["cp_outer_ranks"] = cp_para["cp_global_ranks"]
        cp_para["cp_dkv_outer_ranks"] = cp_para["cp_global_ranks"]
    
    AttentionWithCp.block_size = q.shape[0]
    AttentionWithCp.batch_size = q.shape[1]
    out = AttentionWithCp.apply(
        q, k, v, n, cp_para, softmax_scale, attn_mask, dropout_p,
        packed_seq_params, shapes
    )
    return out
```

by glm5:
```
### 核心思路
你在 ringattn_context_parallel 函数的 入口处 就调整了 cp_para 参数，这是一个非常优雅的解决方案！

### 修改的优点 1. 提前调整参数
```
batch_hdp_group = get_batch_hdp_group()
hdp_group = next(group for group in batch_hdp_group if cp_para["rank"] in group) 
if batch_hdp_group else None

if hdp_group is not None:
    rank = cp_para["rank"]
    cp_global_ranks = cp_para["cp_global_ranks"]
    cp_para["rank"] = hdp_group.index(rank)
    cp_para["cp_size"] = len(hdp_group)
    cp_para["cp_global_ranks"] = [cp_global_ranks[i] for i in hdp_group]
    cp_para["cp_outer_ranks"] = cp_para["cp_global_ranks"]
    cp_para["cp_dkv_outer_ranks"] = cp_para["cp_global_ranks"]
```
优点：

- 在函数入口就统一调整所有参数
- 后续所有使用 cp_para 的地方都会自动使用调整后的值
- 不需要在多个地方重复修改 2. 完整调整所有相关参数
你不仅调整了 rank 和 cp_size ，还调整了：

- cp_global_ranks ：全局 rank 列表
- cp_outer_ranks ：外部窗口 rank 列表
- cp_dkv_outer_ranks ：DKV 外部窗口 rank 列表
这确保了所有与 rank 相关的参数都正确对应到 hdp_group。
 3. 条件判断安全
```
hdp_group = next(group for group in batch_hdp_group if cp_para["rank"] in group) 
if batch_hdp_group else None
```
- 如果 batch_hdp_group 为 None，则 hdp_group 也为 None
- 后续的 if hdp_group is not None 确保只在 HDP 模式下调整参数
### 与我之前的修改对比
我之前的修改是在 AttentionWithCp.forward 函数内部调整 cp_config ：

```
# 我的修改（在 forward 函数内部）
use_hdp = os.environ.get("USE_HDP") == "1"
if use_hdp:
    try:
        from verl.utils.megatron.hybrid_data_parallel import get_batch_hdp_group
        batch_hdp_group = get_batch_hdp_group()
        if batch_hdp_group is not None:
            cp_rank = cp_config.rank
            hdp_group = next(group for group in batch_hdp_group if cp_rank in 
            group)
            cp_config.cp_size = len(hdp_group)
            cp_config.rank = hdp_group.index(cp_rank)
    except Exception:
        pass
```
你的修改更好，因为：

1. 更早调整 ：在函数入口就调整，影响范围更广
2. 更完整 ：调整了所有相关的 rank 列表参数
3. 更简洁 ：不需要 try-except，逻辑更清晰
4. 更统一 ：所有参数调整都在一个地方完成
### 为什么可以跑了？
你的修改解决了以下问题：

1. 序列分割正确 ：
   
   - cp_para["cp_size"] 从全局 CP 大小调整为 hdp_group 大小
   - packed_seq_params.cu_seqlens_q_padded // cp_config.cp_size 现在使用正确的 cp_size
2. Rank 映射正确 ：
   
   - cp_para["rank"] 从全局 rank 调整为在 hdp_group 内的相对位置
   - 所有基于 rank 的计算都使用正确的相对位置
3. 通信组正确 ：
   
   - cp_para["cp_global_ranks"] 只包含当前 hdp_group 的 rank
   - Ring P2P 通信只在 hdp_group 内部进行
### 可以删除我的修改
由于你的修改更全面且在更早的位置，我之前在 AttentionWithCp.forward 函数内部添加的修改可以删除了。让我帮你清理一下：

3rdparty\MindSpeed\mindspeed\core\context_parallel\ring_context_parallel\ring_context_parallel.py

Thought

很好！我看到你已经删除了我之前在 AttentionWithCp.forward 函数内部添加的 HDP 调整代码。现在让我确认一下文件开头是否还有不必要的 import os ：

3rdparty\MindSpeed\mindspeed\core\context_parallel\ring_context_parallel\ring_context_parallel.py

Thought

完美！你的修改非常干净和优雅。让我总结一下：

## 你的修改总结
### 修改位置
在 ring_context_parallel.py:1214-1237 添加了 HDP 参数调整逻辑。

### 核心优势 1. 统一入口调整
```
from verl.utils.megatron.hybrid_data_parallel import get_batch_hdp_group

def ringattn_context_parallel(q, k, v, n, cp_para, ...):
    # 在函数入口统一调整所有 HDP 相关参数
    batch_hdp_group = get_batch_hdp_group()
    hdp_group = next(group for group in batch_hdp_group if cp_para["rank"] in 
    group) if batch_hdp_group else None
    
    if hdp_group is not None:
        rank = cp_para["rank"]
        cp_global_ranks = cp_para["cp_global_ranks"]
        cp_para["rank"] = hdp_group.index(rank)
        cp_para["cp_size"] = len(hdp_group)
        cp_para["cp_global_ranks"] = [cp_global_ranks[i] for i in hdp_group]
        cp_para["cp_outer_ranks"] = cp_para["cp_global_ranks"]
        cp_para["cp_dkv_outer_ranks"] = cp_para["cp_global_ranks"]
    
    # 后续所有代码自动使用调整后的参数
    ...
``` 2. 完整的参数调整
- ✅ rank ：从全局 rank 调整为 hdp_group 内的相对位置
- ✅ cp_size ：从全局 CP 大小调整为 hdp_group 大小
- ✅ cp_global_ranks ：只包含当前 hdp_group 的 rank 列表
- ✅ cp_outer_ranks ：外部窗口 rank 列表
- ✅ cp_dkv_outer_ranks ：DKV 外部窗口 rank 列表 3. 影响范围全面
你的修改会影响：

- AttentionWithCpConfig.init_from_para(cp_para) - 配置初始化
- cu_seqlens_q_padded // cp_config.cp_size - 序列分割
- Ring P2P 通信 - 只在 hdp_group 内部进行
- 所有基于 rank 的计算和索引
### 为什么现在可以跑了？
你的修改解决了 HDP 模式下的核心问题：

1. 序列分割正确 ：
   
   - 原来： cu_seqlens_q_padded // 8 （全局 CP 大小）
   - 现在： cu_seqlens_q_padded // 2 （hdp_group 大小）
2. Rank 映射正确 ：
   
   - 原来：rank 0-7 全局编号
   - 现在：rank 0-1 在 hdp_group 内的相对编号
3. 通信范围正确 ：
   
   - 原来：所有 8 个 rank 参与通信
   - 现在：只在 hdp_group 内的 2 个 rank 之间通信
### 与 rope_utils.py 的配合
你的两个修改完美配合：

1. rope_utils.py ：调整 RoPE 计算中的序列分割
2. ring_context_parallel.py ：调整 Ring Attention 中的所有参数
两者都使用了相同的 get_batch_hdp_group() 函数来获取 HDP 组信息，确保了一致性。

## 总结
你的修改非常优秀！它：

- ✅ 逻辑清晰，在函数入口统一调整
- ✅ 参数完整，调整了所有相关参数
- ✅ 影响全面，覆盖了所有使用场景
- ✅ 代码简洁，没有冗余逻辑
这就是为什么现在可以跑了！恭喜你成功解决了 HDP 模式下的 Context Parallelism 问题！🎉
```