#' Neural Network with Two Layers
#' 
#' This function is a wrapper for a two layered neural network written using the Keras Package. It takes a 
#' @param Text The text that will be used as training and test data.
#' @param Codes The codes that will be used as outcomes to be predicted by the NN model.
#' @param Words The number of top words included in document feature matrixes used as training and testing data.
#' @param Seed The seed used in the model. Defaults to 17
#' @param Weighting The type of feature weighting used in the document feature matrix. I.e., count and tfidf.
#' @param Train_prop The proportion of the data used to train the model. The remainder is used as test data.
#' @param Epochs The number of epochs used in the NN model.
#' @param Units The number of network nodes used in the first layer of the sequential model
#' @param Batch The number of batches estimated
#' @param Dropout A floating variable bound between 0 and 1. It determines the rate at which units are dropped for the linear tranformation of the inputs.
#' @param ValSplit The validation split of the data used in the training of the LSTM model
#' @param Metric Metric used to train algorithm
#' @param Loss Metric used to train algorithm
#' @param CM A logical variable that indicates whether a confusion matrix will be output from the function
#' @param Model A logical variable that indicates whether the trained model should be included in the output of this function
#' @keywords neural networks
#' @export

nn_twolayer <- function(Text, Codes, 
                        Words = 3000, Seed = 17, Weighting = "count", Train_prop = .5, 
                        Epochs = 2, Units = 512, Batch = 32, Dropout = .5, Valsplit = .1,
                        Metric = "accuracy",Loss = "categorical_crossentropy", 
                        CM = TRUE, Model = FALSE){
  set.seed(Seed)
  require(caret)
  require(keras)
  require(dplyr)
  if(length(Text) != length(Codes)) {
    cat("The length of the text and codes variables aren't the same.")
    break
  }
  train_index <- sample(1:length(Text),size = length(Text) * Train_prop, replace = F)
  Codes2 <- as.numeric(as.factor(Codes))
  
  con_train_x <- Text[train_index]
  con_train_y <- Codes2[train_index]
  con_test_x <- Text[-train_index]
  con_test_y <- Codes2[-train_index]
  classes <- length(unique(Codes2)) + 1
  
  if(Words != 0){
    tok <- text_tokenizer(filters = "!\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r",lower = T,num_words = Words) %>% 
      fit_text_tokenizer(con_train_x)
    txt_train <- texts_to_matrix(tok,texts = con_train_x, mode = Weighting)
    txt_test <- texts_to_matrix(tok,texts = con_test_x, mode = Weighting)
  } else {
    tok <- text_tokenizer(filters = "!\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r",lower = T) %>% 
      fit_text_tokenizer(con_train_x)
    txt_train <- texts_to_matrix(tok,texts = con_train_x, mode = Weighting)
    txt_test <- texts_to_matrix(tok,texts = con_test_x, mode = Weighting)
    Words = dim(txt_train)[2]
    cat("There were ",Words," features used to train this model.")
  }
    
  train_y <- to_categorical(as.numeric(con_train_y), num_classes = classes)
  test_y <- to_categorical(as.numeric(con_test_y), num_classes = classes)
  model <- keras_model_sequential() 
  model %>%
    layer_dense(units = Units, input_shape = Words) %>% 
    layer_activation(activation = 'relu') %>% 
    layer_dropout(rate = Dropout) %>% 
    layer_dense(units = classes) %>% 
    layer_activation(activation = 'softmax')
  
  model %>% compile(
    loss = Loss,
    optimizer = 'adam',
    metrics = Metric
  )
  
  history <- model %>% fit(
    txt_train, train_y,
    batch_size = Batch,
    epochs = Epochs,
    verbose = 1,
    validation_split = Valsplit
  )
  score <- model %>% evaluate(
    txt_test, test_y,
    batch_size = Batch,
    verbose = 1
  )
  if(CM == TRUE) {
    pred_class <- predict_classes(model, txt_test, batch_size = Batch)
    score$ConMat <- caret::confusionMatrix(factor(pred_class,
                                                  labels = levels(as.factor(Codes)), 
                                                  levels = 1:length(unique(Codes))),
                                           factor(con_test_y,
                                                  labels = levels(as.factor(Codes)), 
                                                  levels = 1:length(unique(Codes))))
  } 
  if(Model == TRUE) {
    score$Model <- model
  }
  return(score)
}
