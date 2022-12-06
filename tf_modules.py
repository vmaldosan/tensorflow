import tensorflow as tf

class MyModule(tf.Module):
  def __init__(self, value):
    self.weight = tf.Variable(value)

  @tf.function
  def multiply(self, x):
    return x * self.weight

if __name__ == '__main__':
  mod = MyModule(3)
  mod.multiply(tf.constant([1, 2, 3]))
  # print(mod.weight)
  save_path = './saved'
  tf.saved_model.save(mod, save_path)