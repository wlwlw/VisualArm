class AgentConfig(object):
    name = "SimpleQN_Agent"
    model_dir = "./model/SimpleQN_Agent"
    scale = 50
    max_step = 100*scale
    memory_size = 50*scale

    batch_size = 32
    discount = 0.99
    ep_end = 0.1
    ep_start = 1
    ep_end_t = 50*scale
    learn_start_step = 20*scale

    history_length = 2
    train_frequency = 2

    _save_step = max_step/scale
    _test_step = _save_step

class EnvironmentConfig(object):
    env_name = "ArmControlPlateform"
    display=True
    camera_res_x = 640
    camera_res_y = 480
    control_topic = "arm_input_4x3"
    camera_topic = 'redObject_position_and_size_3x1'
    #sensor_topic = "arm_state_4x3"
    observation_space = [7]
    max_reward = 1.
    min_reward = -1.

class DQNConfig(AgentConfig, EnvironmentConfig):
    pass

def get_config(FLAGS):
    if FLAGS.model == 'DQN':
        config = DQNConfig
    else:
        raise TypeError("model config: %s doesn't exist" % (FLAGS.model))

    for k, v in FLAGS.__dict__['__flags'].items():
        if k == 'gpu':
            if v == False:
                config.cnn_format = 'NHWC'
            else:
                config.cnn_format = 'NCHW'

        if hasattr(config, k):
            setattr(config, k, v)

    return config

