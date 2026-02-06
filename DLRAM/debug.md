Traceback (most recent call last):
  File "pretrain.py", line 744, in <module>
    main()
  File "pretrain.py", line 729, in main
    train(
  File "pretrain.py", line 607, in train
    loss_eo = pretrainer.train_step_entity_object(batch_eo)
  File "pretrain.py", line 533, in train_step_entity_object
    negative_image_features = self.model.encode_image(negative_image_inputs)
  File "pretrain.py", line 67, in encode_image
    image_outputs = self.visual_encoder(**image_inputs)
  File "/root/miniconda3/envs/DLRAM/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1051, in _call_impl
    return forward_call(*input, **kwargs)
  File "/root/miniconda3/envs/DLRAM/lib/python3.8/site-packages/transformers/models/vit/modeling_vit.py", line 579, in forward
    encoder_outputs = self.encoder(
  File "/root/miniconda3/envs/DLRAM/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1051, in _call_impl
    return forward_call(*input, **kwargs)
  File "/root/miniconda3/envs/DLRAM/lib/python3.8/site-packages/transformers/models/vit/modeling_vit.py", line 420, in forward
    layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)
  File "/root/miniconda3/envs/DLRAM/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1051, in _call_impl
    return forward_call(*input, **kwargs)
  File "/root/miniconda3/envs/DLRAM/lib/python3.8/site-packages/transformers/models/vit/modeling_vit.py", line 372, in forward
    layer_output = self.intermediate(layer_output)
  File "/root/miniconda3/envs/DLRAM/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1051, in _call_impl
    return forward_call(*input, **kwargs)
  File "/root/miniconda3/envs/DLRAM/lib/python3.8/site-packages/transformers/models/vit/modeling_vit.py", line 319, in forward
    hidden_states = self.dense(hidden_states)
  File "/root/miniconda3/envs/DLRAM/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1051, in _call_impl
    return forward_call(*input, **kwargs)
  File "/root/miniconda3/envs/DLRAM/lib/python3.8/site-packages/torch/nn/modules/linear.py", line 96, in forward
    return F.linear(input, self.weight, self.bias)
  File "/root/miniconda3/envs/DLRAM/lib/python3.8/site-packages/torch/nn/functional.py", line 1847, in linear
    return torch._C._nn.linear(input, weight, bias)
RuntimeError: CUDA out of memory. Tried to allocate 1.08 GiB (GPU 0; 23.57 GiB total capacity; 20.73 GiB already allocated; 596.81 MiB free; 21.45 GiB reserved in total by PyTorch)