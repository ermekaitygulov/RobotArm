# Robot arm project
**Algorithms:**
* DDDQN 
* APEX-DQN/DDPG/TD3 (asynchronus train) implementation with [Ray](https://docs.ray.io/en/latest/)
* DDPG
* TD3 (DDPG with twin critic, with delay updates, with noise in target. It is easier to use TD3 than tune DDPG)

**Implementation details:**
* Two variants of buffer can be used in DQN and DDPG (support of sb implementation
 in APEX is in process):
 
| [ccprb](https://ymd_h.gitlab.io/cpprb/) (C++ realization of replay buffers) | stable_baselines |
|------------|-----------------|
| Fast | Slow|
|Cause high memory usage without memory compress features (next_of; stack_compress and etc.). In some cases they can cause bugs.|Memory effective due to use of LazyFrames in frame_stack and absence of duplicating (next_/n_)state and state|

* Wrapper for cpprb-buffers and stablebaselines buffers
 supports dict observation spaces like:
 ```state = {'pov':{'shape':(64,64,3), 'dtype': 'uint8'}, 'angles': {'shape': (7), 'dtype':'float'}} ```
* Vrep environment for Rozum robot uses [PyRep API](https://github.com/stepjam/PyRep)
 (instead of original VREP API). Rewards uses tolerance function from [DeepMind ControlSuite](https://github.com/deepmind/dm_control)
* Observation type option in environment ('pov'/('pov', 'angles')/'angles' and etc.)
* **If dtype_dict is specified, samplings in algorithms will be wrapped with tf.data.Dataset.from_generator, improving updates frequency**
* There are different make_model functions in algorithms/model.py. 
They can be accessed with get_network_builder(name) function. There is ***_uni** functions that can work with different combinations 
 of obs_spaces in RozumEnv. Depending on space they build CNN/MLP blocks and concatenates them.

**TODO**:
* Unity environment for Rozum model
 to speed up training on server in headless mode