#' Neural Network with a long-term short-term memory and an embedding layer
#' 
#' This function is a wrapper for a long term short term neural network written using the Keras Package.
#'  
#' @param Text The text that will be used as training and test data.
#' @param Codes The codes that will be used as outcomes to be predicted by the NN model.
#' @param Words The number of top words included in document feature matrixes used as training and testing data.
#' @param Seed The seed used in the model. Defaults to 17
#' @param Train_prop The proportion of the data used to train the model. The remainder is used as test data.
#' @param Epochs The number of epochs used in the NN model.
#' @param Batch The number of batches estimated in the NN.
#' @param MaxSentencelen All sentences will be truncated to this length to be input into the LSTM model
#' @param WordEmbedDim The number of word embedding dimensions to be produced by the LSTM model
#' @param ValSplit The validation split of the data used in the training of the LSTM model
#' @param Units_lstm The number of network nodes used in the LSTM layer
#' @param Dropout A floating variable bound between 0 and 1. It determines the rate at which units are dropped for the linear tranformation of the inputs.
#' @param Recurrent_dropout A floating variable bound between 0 and 1. It determines the fraction of the units to drop for the linear transformation of the recurrent neural network layer.
#' @param CM A logical variable that indicates whether a confusion matrix will be output from the function
#' @param Model A logical variable that indicates whether the trained model should be included in the output of this function
#' @keywords neural networks, LSTM
#' @export

nn_embedded_lstm <- function(Text, Codes, 
                             Words = 3000, Seed = 17, Train_prop = 0.5, 
                             Epochs = 10, Batch = 32, MaxSentencelen = 60, WordEmbedDim = 50, ValSplit = 0.1, 
                             Units_lstm = 64, Dropout = 0.2, Recurrent_dropout = 0.2, 
                             CM = TRUE, Model = FALSE) {
  set.seed(Seed)
  train_index <- sample(1:length(Text), size = length(Text) * Train_prop, replace = FALSE)
  Codes2 <- as.numeric(as.factor(Codes))
  
  con_train_x <- Text[train_index]
  con_train_y <- Codes2[train_index]
  con_test_x <- Text[-train_index]
  con_test_y <- Codes2[-train_index]
  classes <- length(unique(Codes2)) + 1
  
  tok <- text_tokenizer(filters = "!\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r", 
                        lower = TRUE, num_words = Words) %>% 
    fit_text_tokenizer(con_train_x)
  
  txt_train <- texts_to_sequences(tok, texts = con_train_x) %>% pad_sequences(maxlen = MaxSentencelen)
  txt_test <- texts_to_sequences(tok, texts = con_test_x) %>% pad_sequences(maxlen = MaxSentencelen)
  train_y <- to_categorical(as.numeric(con_train_y), num_classes = classes)
  test_y <- to_categorical(as.numeric(con_test_y), num_classes = classes)
  
  model <- keras_model_sequential() 
  model %>%
      layer_embedding(input_dim = Words, output_dim = WordEmbedDim, input_length = MaxSentencelen) %>%
      layer_lstm(units = Units_lstm, dropout = Dropout, recurrent_dropout = Recurrent_dropout) %>% 
      layer_dense(units = classes, activation = 'sigmoid')

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
    validation_split = ValSplit
  )
  score <- model %>% evaluate(
    txt_test, test_y,
    batch_size = Batch,
    verbose = 1
  )
  if(CM == TRUE){
      pred_class <- predict_classes(model, txt_test, batch_size = Batch)
      score$ConMat <- caret::confusionMatrix(factor(pred_class,
                                                    labels = levels(as.factor(Codes)), 
                                                    levels = 1:length(unique(Codes))),
                                             factor(con_test_y,
                                                    labels = levels(as.factor(Codes)), 
                                                    levels = 1:length(unique(Codes))))
  } 
  if(Model == TRUE){
      score$Model <- model
  }
  return(score)
}

