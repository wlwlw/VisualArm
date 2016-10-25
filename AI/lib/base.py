import os
import pprint
import inspect

import tensorflow as tf

pp = pprint.PrettyPrinter().pprint

def class_vars(obj):
    return {k:v for k, v in inspect.getmembers(obj)
            if not k.startswith('__') and not callable(k)}

class BaseModel(object):
    """Abstract object representing an Reader model."""
    def __init__(self, config):
        self.config = config

        try:
            self._attrs = config.__dict__['__flags']
        except:
            self._attrs = class_vars(config)
        pp(self._attrs)

        self.config = config

        for attr in self._attrs:
            name = attr if not attr.startswith('_') else attr[1:]
            setattr(self, name, getattr(self.config, attr))

    def save_model(self, filename=None, step=None):
        print(" [*] Saving agent state...")
        if filename is not None:
            self.saver.save(self.sess, filename)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.saver.save(self.sess, os.path.join(self.model_dir,self.name+'-AutoSave'), global_step=step)

    def load_model(self, filename=None):
        print(" [*] Loading agent state...")
        if filename is not None:
            try:
                self.saver.restore(self.sess, filename)
                print(" [*] Load SUCCESS: %s" % filename)
                return True
            except:
                print(" [!] Load FAILED: %s" % filename)
                return False
        ckpt = tf.train.get_checkpoint_state(self.model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            fname = os.path.join(self.model_dir, ckpt_name)
            self.saver.restore(self.sess, fname)
            print(" [*] Load SUCCESS: %s" % fname)
            return True
        else:
            print(" [!] Load FAILED: %s" % self.model_dir)
            return False

    """
    @property
    def model_dir(self):
        model_dir = self.config.env_name
        for k, v in self._attrs.items():
            if not k.startswith('_') and k not in ['display']:
                model_dir += "/%s-%s" % (k, ",".join([str(i) for i in v])
                        if type(v) == list else v)
        return model_dir + '/'
    """
