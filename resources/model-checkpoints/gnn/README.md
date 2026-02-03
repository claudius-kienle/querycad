MF_CAD++_residual_lvl_7_edge_MFCAD++_units_512_date_2021-07-27_epochs_100.ckpt
copied from 
https://gitlab.com/qub_femg/machine-learning/hierarchical-cadnet/-/tree/master/checkpoint?ref_type=heads

with tensorflow 2.5

then converted to tensorflow 2.17 with

```python
# Load your checkpoint
checkpoint = tf.train.Checkpoint(model=model)
checkpoint.restore('path/to/your/checkpoint')

# Save the weights using tf.train.Checkpoint
checkpoint.write('path/to/checkpoint_dir/ckpt')
```