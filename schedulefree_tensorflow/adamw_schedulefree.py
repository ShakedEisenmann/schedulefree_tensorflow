# This file is adapted from the original PyTorch implementation by Meta Platforms, Inc. and affiliates.  
# The corresponding file in the original project can be found at:  
# https://github.com/facebookresearch/schedule_free/blob/main/schedulefree/adamw_schedulefree.py 
#  
# This TensorFlow version was created by Shaked Eisenmann in 2025.  
#  
# This source code is licensed under the license found in the LICENSE file  
# located in the root directory of this source tree.  

import tensorflow as tf

class AdamWScheduleFree(tf.keras.optimizers.Optimizer):
    """
    Schedule-Free AdamW
    
    This is a TensorFlow implementation of the Schedule-Free AdamW optimizer,  
    originally developed by Facebook Research.

    As the name suggests, no scheduler is needed with this optimizer.
    To add warmup, rather than using a learning rate schedule you can just
    set the warmup_steps parameter.

    This optimizer requires that .train() and .eval() be called before the
    beginning of training and evaluation respectively. The optimizer should
    also be placed in eval mode when saving checkpoints.

    For more details and the original PyTorch implementation, see:  
    https://github.com/facebookresearch/schedule_free/
    """

    def __init__(self, learning_rate=0.0025, betas=(0.9, 0.999), weight_decay=0, warmup_steps=0, r=0.0, weight_lr_power=2.0, **kwargs):
        super().__init__(name="AdamWScheduleFree", weight_decay=weight_decay, **kwargs)
        self._learning_rate = self._build_learning_rate(learning_rate)
        self.betas = betas
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.r = r
        self.weight_lr_power = weight_lr_power

    def eval(self, var_list):
        """
        Prepares the optimizer for evaluation by setting variables to their corresponding x values,  
        as described in the original PyTorch implementation.

        This method must be called at the beginning of evaluation, before running test steps.
        """
        if not hasattr(self, "_built") or not self._built or not self.train_mode:
            return
        for variable in var_list:
            var_key = self._var_key(variable)
            i = self._index_dict[var_key]
            variable.assign_add((self.z_variables[i] - variable) * (1 - 1 / self.betas[0])) # Set variable to x
        self.train_mode.assign(False)

    def train(self, var_list):
        """
        Prepares the optimizer for training by setting variables to their corresponding y values,  
        as described in the original PyTorch implementation.

        This method must be called at the beginning of each epoch, before running train steps.
        """
        if not hasattr(self, "_built") or not self._built or self.train_mode:
            return
        for variable in var_list:
            var_key = self._var_key(variable)
            i = self._index_dict[var_key]
            variable.assign_add((self.z_variables[i] - variable) * (1 - self.betas[0])) # Set variable to y
        self.train_mode.assign(True)

    def build(self, var_list):
        """
        Initialize optimizer variables.
        """
        super().build(var_list)
        if hasattr(self, "_built") and self._built:
            return
        self.z_variables = []
        self.weight_sum = []
        self.exp_avg_sq = []
        for var in var_list:
            self.z_variables.append(self.add_variable_from_reference(model_variable=var, variable_name="z", initial_value=var))
            self.weight_sum.append(self.add_variable_from_reference(model_variable=var, variable_name="weight_sum", initial_value=0))
            self.exp_avg_sq.append(self.add_variable_from_reference(model_variable=var, variable_name="exp_avg_sq", initial_value=tf.zeros_like(var)))

        self.lr_max = tf.Variable(-1.0, trainable=False)
        self.train_mode = tf.Variable(True, trainable=False)
        self._built = True

    def update_step(self, gradient, variable):
        """
        Applies the gradient to the associated model variable to update it.
        """

        if not self.train_mode:
            raise Exception("Optimizer was not in train mode when update_step is called. "
                            "Please insert .train() and .eval() calls on the "
                            "optimizer. See https://github.com/facebookresearch/schedule_free/ for details.")

        i = self._index_dict[self._var_key(variable)]
        z = self.z_variables[i]
        lr = self.learning_rate
        k = tf.cast(self.iterations, variable.dtype)

        sched = tf.cond(k < self.warmup_steps, lambda: (k + 1) / self.warmup_steps, lambda: 1.0)
        bias_correction2 = 1 - self.betas[1] ** (k + 1)
        lr = lr * sched * tf.math.sqrt(bias_correction2)
        self.lr_max.assign(tf.maximum(lr, self.lr_max))

        weight = ((k + 1) ** self.r) * (self.lr_max**self.weight_lr_power)
        weight_sum = self.weight_sum[i]
        weight_sum.assign_add(weight)
        ckp1 = tf.math.divide_no_nan(weight, weight_sum)

        exp_avg_sq = self.exp_avg_sq[i]
        exp_avg_sq.assign(exp_avg_sq * self.betas[1] + (gradient**2) * (1 - self.betas[1]))
        denom = tf.math.sqrt(exp_avg_sq) + tf.keras.backend.epsilon()

        # Reuse grad buffer for memory efficiency
        grad_normalized  = tf.math.divide_no_nan(gradient, denom)

        # Weight decay calculated at y
        grad_normalized  = tf.cond(self.weight_decay != 0, lambda: grad_normalized  + variable * self.weight_decay, lambda: grad_normalized )

        # These operations update y in-place,
        # without computing x explicitly.
        variable.assign_add((z - variable) * (ckp1))
        variable.assign_add(grad_normalized  * lr * (self.betas[0] * (1 - ckp1) - 1))

        # z step
        z.assign_sub(lr * grad_normalized )
