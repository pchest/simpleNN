#' Neural Network with Two Layers
#' 
#' This function is a wrapper for a two layered neural network written using the Keras Package. It takes a 
#' @param Words The number of top words included in document feature matrixes used as training and testing data.
#' @param Text The text that will be used as training and test data.
#' @param Codes The codes that will be used as outcomes to be predicted by the NN model.
#' @param Epochs The number of epochs used in the NN model.
#' @param Weighting The type of feature weighting used in the document feature matrix. I.e., count and tfidf.
#' @param Seed The seed used in the model. Defaults to 17
#' @param Batch The number of batches estimated in the NN.
#' @keywords neural networks
#' @export
#' nn_twolayer()

nn_twolayer <- function(Words = 3000, Text = man_dat2$text, Codes = man_dat2$cmp_code2, Epochs = 2, Weighting = "count", Seed = 17, Batch = 32){
  set.seed(Seed)
  if(length(Text != length(Codes))){
    cat("The length of the text and codes variables aren't the same.")
    break
  }
#  man_dat_sub <- sample_n(Data,size = Sample,replace = F)
  train_index <- sample(1:length(Text),size = length(Text)*.5,replace = F)
  Codes2 <- as.numeric(as.factor(Codes))
  con_train_x <- Text[train_index]
  con_train_y <- Codes2[train_index]
  con_test_x <- Text[-train_index]
  con_test_y <- Codes2[-train_index]
  classes <- length(unique(Codes))+1
  tok <- text_tokenizer(filters = "!\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r",lower = T,num_words = Words) %>% 
    fit_text_tokenizer(con_train_x)
  txt_train <- texts_to_matrix(tok,texts = con_train_x, mode = "count")
  txt_test <- texts_to_matrix(tok,texts = con_test_x, mode = "count")
  length(unique(con_train_y)) ; length(unique(con_test_y))
  train_y <- to_categorical(as.numeric(con_train_y),num_classes = classes)
  test_y <- to_categorical(as.numeric(con_test_y), num_classes = classes)
  
  model <- keras_model_sequential() 
  model %>%
    layer_dense(units = 512, input_shape = Words) %>% 
    layer_activation(activation = 'relu') %>% 
    layer_dropout(rate = 0.5) %>% 
    layer_dense(units = classes) %>% 
    layer_activation(activation = 'softmax')
  
  model %>% compile(
    loss = 'categorical_crossentropy',
    optimizer = 'adam',
    metrics = c('accuracy')
  )
  
  history <- model %>% fit(
    txt_train, train_y,
    batch_size = Batch,
    epochs = Epochs,
    verbose = 1,
    validation_split = 0.1
  )
  score <- model %>% evaluate(
    txt_test, test_y,
    batch_size = Batch,
    verbose = 1
  )
  return(score)
}

