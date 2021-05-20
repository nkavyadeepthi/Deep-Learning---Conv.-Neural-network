library(keras)
library(reticulate)
library(tidyverse)
mnist <- keras::dataset_mnist()
str(mnist)

x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

digit <- x_train[10,,]
plot(as.raster(digit, max = 255))
y_train[10]


9000
30 * 60 * 5
3

#(samples, frames, height, width, channels)
1 * 9000 * 1920 * 1080 * 3
#1920×1080
#192×108
#1920 * 1080
#300 * 300

#(samples, kHz, value)


c(c(x_train, y_train), c(x_test, y_test)) %<-% mnist
str(x_train)


## Reshape x from 3d to 2d (by default, row-major)
x_train <- array_reshape(x_train, c(nrow(x_train), 784))
x_test <- array_reshape(x_test, c(nrow(x_test), 784))
str(x_train) # View(x_train[1,])


## Rescale values so that they are between 0 and 1
x_train <- x_train / 255
x_test <- x_test / 255
# View(x_train[1,]) 
## Prepare the y data
## The y data is an integer vector with values ranging from 0 to 9
y_train[1:5]

to_categorical(0, 10)
to_categorical(1, 10)
to_categorical(9, 10)

## Deep learning models prefer binary values rather than integers
y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)
str(y_train) # View(y_train)

x = -1
max(c(0,x))


x = 8
max(c(0,x))
to_categorical(1, 10)

c(0.9, 0.1, 0.2, 0.3, 0.7, 0.4, 0.1, 0.3, 0.2, 0.1)


## Sequential model is simply a linear stack of layers
model <- keras_model_sequential() 
## Define the structure of the neural net
model %>% # A dense layer is a fully connected layer
  layer_dense(units = 256, activation = 'relu', input_shape = c(784)) %>% 
  layer_dropout(rate = 0.4) %>% # randomly set 40% of weights to 0
  layer_dense(units = 128, activation = 'relu') %>% 
  layer_dropout(rate = 0.3) %>% # this helps prevent overfitting
  layer_dense(units = 10, activation = 'softmax') # probability of each class
summary(model)


## Compile the model with appropriate loss function, optimizer, and metrics
model %>% compile(
  optimizer = optimizer_rmsprop(),      # see next slide
  loss = 'categorical_crossentropy',    # since we have 10 categoreis
  metrics = c('accuracy')               # for classification
)

## Use x_train and y_train for training
history <- model %>% fit(
  x_train, y_train, 
  batch_size = 128,      # a set of 128 samples
  epochs = 30,           # let's go through x_train 30 times
  validation_split = 0.2 # use the last 20% of train data for validation
)
plot(history)

