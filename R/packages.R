#' Packages: a function for loading packages
#' 
#' A function designed to simplify loading and installing functions
#' @param lib A string character containing the names of CRAN packages to be loaded or installed in R
#' @export

packages <- function(lib = NA){
  for(l in lib){
    if(!require(l,character.only = T)){
      install.packages(l)
      require(l,character.only = T)
    } else{
      require(l,character.only = T)
    }
  }
}