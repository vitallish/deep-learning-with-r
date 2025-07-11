library(tensorflow)
library(keras)
mnist <- dataset_mnist()

train_images <- mnist$train_$x
train_labels <- mnist$train_$y
test_images <- mnist$test_$x
test_labels <- mnist$test_$y

model <- keras_model_sequential(list(
  layer_dense(units = 512, activation = "relu"),
  layer_dense(units = 10, activation = "softmax")
))

model$compile(
  optimizer = "rmsprop",
  loss = "sparse_categorical_crossentropy",
  metrics = list("accuracy")
)
