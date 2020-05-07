# Robot arm project
**Done**:
* DDDQN (use tf.data to speed up updates)
* APEX-DQN (asynchronus DQN) implementation with [Ray](https://docs.ray.io/en/latest/)
* DDPG for continues control
* Use of [ccprb](https://ymd_h.gitlab.io/cpprb/) (C++ realization of replay buffers) 
* Wrapper for cpprb-buffers and stablebaselines buffers
 supports dict observation spaces
* PyRep environment for Rozum robot

**TODO**:
* APEX-DDPG (asynchronus DDPG)
* Unity environment for Rozum model
 to speed up training on server in headless mode

     