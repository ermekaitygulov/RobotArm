# Robot arm project
**Algorithms:**
* DDDQN 
* APEX-DQN (asynchronus DQN) implementation with [Ray](https://docs.ray.io/en/latest/)
* DDPG for continues control

**Implementation details:**
* Two variants of buffer can be used:
 
| [ccprb](https://ymd_h.gitlab.io/cpprb/) (C++ realization of replay buffers) | stable_baselines |
|------------|-----------------|
| Fast | Slow|
|Cause high memory usage without memory compress features (next_of; stack_compress and etc.). In some cases they can cause bugs.|Memory effective due to use of LazyFrames in frame_stack and absence of duplicating (next_/n_)state and state|

* Wrapper for cpprb-buffers and stablebaselines buffers
 supports dict observation spaces like:
 ```state = {'pov':{'shape':(64,64,3), 'dtype': 'uint8'}, 'angles': {'shape': (7), 'dtype':'float'}} ```
* PyRep environment for Rozum robot (with rewards from DeepMind ControlSuite)

**TODO**:
* APEX-DDPG (asynchronus DDPG)
* Observation type option in environment ('pov'/('pov', 'angles')/'angles' and etc.)
* Unity environment for Rozum model
 to speed up training on server in headless mode

**InProc**:
* alg configs support:
    * [X] DQN
    * [ ] DDPG
    * [ ] APEX-DQN    